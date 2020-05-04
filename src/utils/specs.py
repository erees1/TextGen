import yaml

def load_spec(spec_path):

    if spec_path is not None:
        pass
    else:
        print('Must provide spec path')
        exit(1)

    with open(spec_path) as f:
        specs = yaml.load(f, Loader=yaml.Loader)

    parsed_specs = {}
    for param_type in specs.keys():
        params = specs[param_type]
        parsed_params = {}
        for key, value in params.items():
            try:
                parsed_params[key] = int(value)
            except Exception:
                parsed_params[key] = value
        parsed_specs[param_type] = parsed_params

    return parsed_specs
