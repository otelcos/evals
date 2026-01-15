# Open Telco CLI Architecture

## Screen Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENTRY POINT                             │
│                       satellite command                         │
│                    (pyproject.toml script)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OpenTelcoApp                             │
│                      (cli/app.py)                               │
│                                                                 │
│  - Main Textual App class                                       │
│  - Manages screen stack                                         │
│  - Pushes WelcomeScreen on mount                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       WelcomeScreen                             │
│                  (cli/screens/welcome.py)                       │
│                                                                 │
│  ┌────────────────────────────────────┐                         │
│  │  ✱ Welcome to Open Telco           │                         │
│  │                                    │                         │
│  │  ██████╗ ██████╗ ███████╗███╗...   │                         │
│  │  (ASCII ART LOGO)                  │                         │
│  │                                    │                         │
│  │  Press Enter to continue           │                         │
│  └────────────────────────────────────┘                         │
│                                                                 │
│  Bindings:                                                      │
│    [Enter] → switch_screen(MainMenuScreen)                      │
│    [Escape] → switch_screen(MainMenuScreen)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                          Enter/Escape
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MainMenuScreen                            │
│                 (cli/screens/main_menu.py)                      │
│                                                                 │
│  ┌────────────────────────────────────┐                         │
│  │         Main Menu                  │                         │
│  │  ┌──────────────────────────────┐  │                         │
│  │  │  1. Set Models               │  │                         │
│  │  │  2. Run Evals                │  │                         │
│  │  │  3. Preview Results          │  │                         │
│  │  │  4. Submit                   │  │                         │
│  │  └──────────────────────────────┘  │                         │
│  │  Press Enter to select, Q to quit  │                         │
│  └────────────────────────────────────┘                         │
│                                                                 │
│  Bindings:                                                      │
│    [↑/↓] → Navigate menu items                                  │
│    [Enter] → Select item (shows notification)                   │
│    [Q] → app.exit()                                             │
│    [Escape] → app.exit()                                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                           Q/Escape
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                           EXIT                                  │
│                    Return to terminal                           │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/evals/cli/
├── __init__.py          # Package init, exports main()
├── app.py               # OpenTelcoApp (main Textual App)
├── diagram.md           # This file
└── screens/
    ├── __init__.py      # Exports all screen classes
    ├── welcome.py       # WelcomeScreen (ASCII art splash)
    └── main_menu.py     # MainMenuScreen (4 options)
```

## Key Components

### Entry Point (`cli/__init__.py`)
```python
def main() -> None:
    app = OpenTelcoApp()
    app.run()
```

### App Class (`cli/app.py`)
- Extends `textual.app.App`
- Pushes `WelcomeScreen` on mount
- Manages screen transitions

### Screen Classes
Each screen extends `textual.screen.Screen` and defines:
- `BINDINGS` - Key bindings for user actions
- `DEFAULT_CSS` - Styling for the screen
- `compose()` - Widget layout
- Action methods - Handle user interactions

## Menu Options (Future Implementation)

| Option | ID | Purpose |
|--------|-----|---------|
| Set Models | `set-models` | Configure AI models for evaluation |
| Run Evals | `run-evals` | Execute benchmark evaluations |
| Preview Results | `preview-results` | View evaluation results |
| Submit | `submit` | Submit results to leaderboard |
