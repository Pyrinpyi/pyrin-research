use clap::{Parser, Subcommand};

#[derive(Clone, Parser)]
#[command(version, about, long_about = None)]
pub struct Cli
{
    /// Optional name to operate on
    name: Option<String>,

    // /// Sets a custom config file
    // #[arg(short, long, value_name = "FILE")]
    // config: Option<PathBuf>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub debug: u8,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Clone, Subcommand)]
pub enum Commands
{
    /// does testing things
    Test {
        /// lists test values
        #[arg(short, long)]
        list: bool,
    },
}

pub fn setup() -> Cli
{
    let cli = Cli::parse();

    // You can check the value provided by positional arguments, or option arguments
    if let Some(name) = cli.name.as_deref()
    {
        println!("Value for name: {name}");
    }

    // if let Some(config_path) = cli.config.as_deref() {
    //     println!("Value for config: {}", config_path.display());
    // }

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    match &cli.command
    {
        Some(Commands::Test { list }) =>
        {
            if *list
            {
                println!("Printing testing lists...");
            }
            else
            {
                println!("Not printing testing lists...");
            }
        }
        None =>
        {
        }
    }

    return cli;
}