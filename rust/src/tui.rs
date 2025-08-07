use std::io;
use std::io::{stdout, Stdout};
use std::time::Duration;

use ratatui::layout::{Constraint, Layout};
use ratatui::style::Style;
use ratatui::widgets::{Bar, BarChart, BarGroup, Padding};
use ratatui::{
    backend::CrosstermBackend,
    buffer::Buffer,
    crossterm::{
        event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    },
    layout::{Alignment, Rect},
    style::Stylize,
    symbols::border,
    text::Line,
    widgets::{block::Title, Block, Paragraph, Widget},
    Terminal,
};

use crate::zootopia::{Pos, TerminalState};
use crate::interactive_play::{InteractivePlay, Snapshot};
use crate::types::{EvalPosT, QValue};

/// A type alias for the terminal type used in this application
pub type Tui = Terminal<CrosstermBackend<Stdout>>;

/// Initialize the terminal
pub fn init() -> io::Result<Tui> {
    execute!(stdout(), EnterAlternateScreen)?;
    enable_raw_mode()?;
    Terminal::new(CrosstermBackend::new(stdout()))
}

/// Restore the terminal to its original state
pub fn restore() -> io::Result<()> {
    execute!(stdout(), LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}

#[derive(Debug)]
pub struct App<E: EvalPosT> {
    game: InteractivePlay<E>,
    exit: bool,
}

impl<E: EvalPosT + Send + Sync + 'static> App<E> {
    pub fn new(
        eval_pos: E,
        max_mcts_iterations: usize,
        c_exploration: f32,
        c_ply_penalty: f32,
    ) -> Self {
        Self {
            game: InteractivePlay::new(eval_pos, max_mcts_iterations, c_exploration, c_ply_penalty),
            exit: false,
        }
    }

    /// runs the application's main loop until the user quits
    pub fn run(&mut self, terminal: &mut Tui) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| {
                let snapshot = self.game.snapshot();
                draw_app(&snapshot, frame.size(), frame.buffer_mut());
            })?;

            if event::poll(Duration::from_millis(100))? {
                self.handle_events()?;
            }
        }
        Ok(())
    }

    /// updates the application's state based on user input
    fn handle_events(&mut self) -> io::Result<()> {
        match event::read()? {
            // it's important to check that the event is a key press event as
            // crossterm also emits key release and repeat events on Windows.
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            _ => {}
        };
        Ok(())
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('b') => self.game.make_random_move(0.0),
            KeyCode::Char('m') => self.game.increase_mcts_iters(100),
            KeyCode::Char('n') => self.game.reset_game(),
            KeyCode::Char('r') => self.game.make_random_move(1.0),
            KeyCode::Char('q') => self.exit = true,
            KeyCode::Char('t') => self.game.increase_mcts_iters(1),
            KeyCode::Char('u') => self.game.undo_move(),
            // Directional movement controls for Zootopia
            KeyCode::Up | KeyCode::Char('w') => self.game.make_move(crate::zootopia::Move::Up),
            KeyCode::Down | KeyCode::Char('s') => self.game.make_move(crate::zootopia::Move::Down),
            KeyCode::Left | KeyCode::Char('a') => self.game.make_move(crate::zootopia::Move::Left),
            KeyCode::Right | KeyCode::Char('d') => self.game.make_move(crate::zootopia::Move::Right),
            _ => {}
        };
    }
}

fn draw_app(snapshot: &Snapshot, rect: Rect, buf: &mut Buffer) {
    let title = Title::from(" zta0 - Zootopia Alpha-Zero ".bold());
    let outer_block = Block::bordered()
        .title(title.alignment(Alignment::Center))
        .padding(Padding::horizontal(1))
        .border_set(border::THICK);
    let inner = outer_block.inner(rect);
    outer_block.render(rect, buf);

    let layout = Layout::vertical([
        Constraint::Length(24), // Game, Evals
        Constraint::Fill(1),    // Policy
        Constraint::Length(11), // Instructions
    ])
    .spacing(1)
    .split(inner);

    draw_game_and_evals(&snapshot, layout[0], buf);
    draw_policy(&snapshot, layout[1], buf);
    draw_instructions(layout[2], buf);
}

fn draw_game_and_evals(snapshot: &Snapshot, rect: Rect, buf: &mut Buffer) {
    let _isp0 = snapshot.pos.ply() % 2 == 0;
    let to_play = match snapshot.pos.is_terminal_state() {
        Some(TerminalState::Success) => vec![" Animal".green(), " escaped!".into()],
        Some(TerminalState::Failure) => vec![" Caught by".red(), " zookeeper!".into()],
        Some(TerminalState::Timeout) => vec![" Game".gray(), " timeout".into()],
        Some(TerminalState::InProgress) => vec![" Animal".green(), " turn".into()],
        None => vec![" Animal".green(), " turn".into()],
    };

    let block = Block::bordered()
        .title(" Game")
        .title_bottom(to_play)
        .padding(Padding::uniform(1));
    let inner = block.inner(rect);
    block.render(rect, buf);

    let layout = Layout::horizontal([Constraint::Length(40), Constraint::Length(18)])
        .spacing(1)
        .split(inner);

    draw_game_grid(&snapshot.pos, layout[0], buf);
    draw_evals(snapshot.q_penalty, snapshot.q_no_penalty, layout[1], buf);
}

fn draw_game_grid(pos: &Pos, rect: Rect, buf: &mut Buffer) {
    let cell_width = 3;
    let cell_height = 1;
    let (width, height) = pos.dimensions();
    for y in 0..height {
        for x in 0..width {
            let cell_rect = Rect::new(
                rect.left() + (x as u16 * cell_width),
                rect.top() + (y as u16 * cell_height),
                cell_width,
                cell_height,
            )
            .intersection(rect);
            let cell_char = if (x, y) == pos.player_position() {
                'P'
            } else {
                match pos.get_cell_content(x, y) {
                    Some(crate::zootopia::CellContent::Empty) => ' ',
                    Some(crate::zootopia::CellContent::Wall) => '#',
                    Some(crate::zootopia::CellContent::Pellet) => '•',
                    Some(crate::zootopia::CellContent::ZookeeperSpawn) => 'Z',
                    Some(crate::zootopia::CellContent::AnimalSpawn) => 'A',
                    Some(crate::zootopia::CellContent::PowerPellet) => 'O',
                    Some(crate::zootopia::CellContent::ChameleonCloak) => 'C',
                    Some(crate::zootopia::CellContent::Scavenger) => 'S',
                    Some(crate::zootopia::CellContent::BigMooseJuice) => 'M',
                    None => '?',
                }
            };
            Paragraph::new(cell_char.to_string())
                .centered()
                .render(cell_rect, buf);
        }
    }
}

fn draw_evals(q_penalty: QValue, q_no_penalty: QValue, rect: Rect, buf: &mut Buffer) {
    let value_max = 1000u64;
    let q_penalty_u64 = ((q_penalty + 1.0) / 2.0 * (value_max as f32)) as u64;
    let q_no_penalty_u64 = ((q_no_penalty + 1.0) / 2.0 * (value_max as f32)) as u64;
    let bars = vec![
        Bar::default()
            .label("Eval".into())
            .value(q_penalty_u64)
            .text_value(format!("{:.2}", q_penalty).into())
            .style(if q_penalty >= 0.0 {
                Style::new().red()
            } else {
                Style::new().blue()
            }),
        Bar::default()
            .label("Win %".into())
            .value(q_no_penalty_u64)
            .text_value(format!("{:.0}%", q_no_penalty * 100.).into())
            .style(if q_no_penalty >= 0.0 {
                Style::new().red()
            } else {
                Style::new().blue()
            }),
    ];
    BarChart::default()
        .data(BarGroup::default().bars(&bars))
        .bar_width((rect.width - 4) / 2 - 1)
        .bar_gap(2)
        .max(value_max)
        .value_style(Style::new().green().bold())
        .label_style(Style::new().white())
        .render(rect, buf);
}

fn draw_policy(snapshot: &Snapshot, rect: Rect, buf: &mut Buffer) {
    let mcts_status = Line::from(vec![
        " ".into(),
        if snapshot.bg_thread_running {
            "MCTS running: ".green()
        } else {
            "MCTS stopped: ".red()
        },
        snapshot.n_mcts_iterations.to_string().bold(),
        "/".into(),
        snapshot.max_mcts_iterations.to_string().bold(),
    ]);

    let policy_max = 1000u64;
    let move_labels = ["↑", "↓", "←", "→"]; // Up, Down, Left, Right
    let bars = snapshot
        .policy
        .iter()
        .enumerate()
        .map(|(i, p)| {
            Bar::default()
                .label(move_labels[i].into())
                .value((p * (policy_max as f32)) as u64)
                .text_value(format!("{:.2}", p)[1..].into())
        })
        .collect::<Vec<_>>();

    BarChart::default()
        .data(BarGroup::default().bars(&bars))
        .bar_width(5)
        .bar_gap(2)
        .max(policy_max)
        .bar_style(Style::new().yellow())
        .value_style(Style::new().green().bold())
        .label_style(Style::new().white())
        .block(
            Block::bordered()
                .title(" Policy")
                .title_bottom(mcts_status)
                .padding(Padding::uniform(1)),
        )
        .render(rect, buf);
}

fn draw_instructions(rect: Rect, buf: &mut Buffer) {
    let instruction_text = vec![
        Line::from(vec!["<Arrow Keys/WASD>".blue().bold(), " Move animal".into()]),
        Line::from(vec!["<B>".blue().bold(), " Play the best move".into()]),
        Line::from(vec!["<R>".blue().bold(), " Play a random move".into()]),
        Line::from(vec!["<M>".blue().bold(), " More MCTS iterations".into()]),
        Line::from(vec!["<U>".blue().bold(), " Undo last move".into()]),
        Line::from(vec!["<N>".blue().bold(), " New game".into()]),
        Line::from(vec!["<Q>".blue().bold(), " Quit".into()]),
    ];
    Paragraph::new(instruction_text)
        .block(
            Block::bordered()
                .title(" Instructions")
                .padding(Padding::uniform(1)),
        )
        .render(rect, buf);
}
