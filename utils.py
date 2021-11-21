import csv
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import wandb
import yaml

class ExpHandler:
    _home = Path.home()

    def __init__(self, en_wandb=False):
        project_name = os.getenv('project_name', default='default_project')
        exp_name = os.getenv('exp_name', default='default_group')
        run_name = os.getenv('run_name', default='default_name')
        self._exp_id = f'{self._get_exp_id()}_{run_name}'
        self._exp_name = exp_name

        self._save_dir = os.path.join('{}/.exp/{}'.format(self._home, os.getenv('WANDB_PROJECT', default='default_project')),
                                        exp_name, self._exp_id)
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        sym_dest = self._get_sym_path('N')
        os.symlink(self._save_dir, sym_dest)

        self._logger = self._init_logger()
        self._en_wandb = en_wandb

        if en_wandb:
            wandb.init(project = project_name, group=exp_name, name=run_name)

    def _get_sym_path(self, state):
        sym_dir = f'{self._home}/.exp/syms'
        if not os.path.exists(sym_dir):
            os.makedirs(sym_dir)

        sym_dest = os.path.join(sym_dir, '--'.join([self._exp_id, state, self._exp_name]))
        return sym_dest

    @property
    def save_dir(self):
        return self._save_dir

    @staticmethod
    def _get_exp_id():
        with open(f'{ExpHandler._home}/.core/counter', 'r+') as f:
            counter = eval(f.read())
            f.seek(0)
            f.write(str(counter + 1))
        with open(f'{ExpHandler._home}/.core/identifier', 'r+') as f:
            identifier = f.read()[0]
        exp_id = '{}{:04d}'.format(identifier, counter)
        return exp_id

    def _init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(os.path.join(self._save_dir, f'{self._exp_id}_log.txt'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def save_config(self, args):
        conf = vars(args)
        conf['exp_id'] = self._exp_id
        conf['commit'] = os.getenv('commit', default='not_set')
        with open(f'{self._save_dir}/{self._exp_id}_config.yaml', 'w') as f:
            yaml.dump(conf, f)

        if self._en_wandb:
            wandb.config.update(conf)

    def write(self, epoch, eval_metrics=None, train_metrics=None, **kwargs):
        rowd = OrderedDict(epoch=epoch)
        rowd.update(kwargs)
        if eval_metrics:
            rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        if train_metrics:
            rowd.update([('train_' + k, v) for k, v in train_metrics.items()])

        path = os.path.join(self._save_dir, f'{self._exp_id}_summary.csv')
        initial = not os.path.exists(path)
        with open(path, mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=rowd.keys())
            if initial:
                dw.writeheader()
            dw.writerow(rowd)

        if self._en_wandb:
            wandb.log(rowd)

    def log(self, msg):
        self._logger.info(msg)

    def finish(self):
        Path(f'{self._save_dir}/finished').touch()
        os.rename(self._get_sym_path('N'), self._get_sym_path('Y'))
