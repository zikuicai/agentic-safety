import abc
import datetime
import os.path
import time
import json


class Simulator(abc.ABC):
    def __init__(self, cfg, logs_fname: str = None):
        self.cfg = cfg

        self.logs_dir = cfg.logs_dir
        logs_fname = logs_fname if logs_fname is not None else str(datetime.datetime.now()).replace(' ',
                                                                                                    '-') + '.log'
        self.logs_file_path = os.path.join(self.logs_dir, logs_fname)
        os.makedirs(self.logs_dir, exist_ok=True)
        if os.path.exists(self.logs_file_path):
            os.remove(self.logs_file_path)

    def _log_to_file(self, role: str, data, cost):
        data = {'role': role, 'data': data, 'cost': cost, 'timestamp': time.time()}
        if os.path.exists(self.logs_file_path):
            with open(self.logs_file_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(data)

        with open(self.logs_file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)

    def _log_response(self, role: str, response_content, cost):
        self._log_to_file(role, response_content, cost)
