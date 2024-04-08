import math

def parse_spec(spec_str)-> list[str]:
    return [s.strip() for s in spec_str.split(",")]

def collect_data_per_spec(spec_str, pd_row, return_decoded_spec=False):
    columns = parse_spec(spec_str)
    data = []
    for id_col in columns:
        data.append(str(pd_row[id_col]))
    
    output = data if not return_decoded_spec else (data, columns)

    return output

def make_id(id_spec:str, pd_row):
    data = collect_data_per_spec(id_spec, pd_row)

    data = [str(d) for d in data]
    id_str = '_'.join(data)
    return id_str

def make_src(src_spec:str, pd_row):
    data, spec_keys = collect_data_per_spec(src_spec, pd_row, return_decoded_spec=True)

    src_str = []
    for spec_key, d in zip(spec_keys, data):
        # src_str.append(f"{spec_key}: {d}")
        src_str.append(f"{d}")
    src_str = '\n'.join(src_str)
    return src_str

def make_df_id_column(row, id_spec:str):
    row['id'] = make_id(id_spec, row)
    return row

def make_df_src_column(row, src_spec: str):
    row['src'] = make_src(src_spec, row)
    return row

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr


if __name__ == "__main__":

    # Example usage:
    num_rounds = 300
    initial_lr = 5e-5
    min_lr = 1e-6

    lr_list = []
    for round in range(num_rounds):
        lr = cosine_learning_rate(round, num_rounds, initial_lr, min_lr)
        lr_list.append(lr)
        print(f"Round {round + 1}/{num_rounds}, Learning Rate: {lr:.8f}")
