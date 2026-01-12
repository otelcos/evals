from inspect_ai import eval_set

success, logs = eval_set(
    tasks=[
        "telelogs/telelogs.py",
        "three_gpp/three_gpp.py",
        "teleqna/teleqna.py",
        "telemath/telemath.py",
    ],  # set how many tasks you want to run
    model=[
        "openrouter/openai/gpt-5.2",
        "openrouter/anthropic/claude-opus-4.5",
        "openrouter/google/gemini-3-flash-preview",
        "openrouter/mistralai/mistral-large-3-2512",
        "openrouter/deepseek/deepseek-v3.2",
    ],  # set models you want to run in parallel
    log_dir="logs/leaderboard",  # set directory
    epochs=1,  # set resampling iterations
    temperature=0.0,
)


# for more information: https://inspect.aisi.org.uk/eval-sets.html
