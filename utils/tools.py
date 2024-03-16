def is_main_process(script_args):
    return script_args.local_rank<=0