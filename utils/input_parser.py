class InputParser:
    def __init__(self):
        pass

    @staticmethod
    def get_series_from_str_separated_by_dash(dir_str):
        results = []
        dirs = dir_str.split('-')
        for i in range(int(dirs[0]), int(dirs[1]) + 1):
            results.append(str(i))
        return results

    @staticmethod
    def parse_input_string(dir_str):
        results = []
        dirs = dir_str.split(',')
        for tmp_dir in dirs:
            if '-' in tmp_dir:
                for tmp in InputParser.get_series_from_str_separated_by_dash(tmp_dir):
                    results.append(tmp)
            else:
                results.append(tmp_dir)
        return results
