from inspect_ai import eval_set

success, logs = eval_set(
   tasks=["telelogs/telelogs.py", "three_gpp/three_gpp.py", "teleqna/teleqna.py", "telemath/telemath.py"], # set how many tasks you want to run
   model=["openrouter/openai/gpt-4o", "openrouter/openai/gpt-3.5-turbo"], # set models you want to run in parallel
   log_dir="logs/logs-run-1", # set directory
   limit=1, # set how many samples you want to run
   epochs=1, # set resampling iterations
)


# for more information: https://inspect.aisi.org.uk/eval-sets.html