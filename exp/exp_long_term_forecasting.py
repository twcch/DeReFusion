from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils import recovery
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _count_parameters(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def _reset_gpu_peak_memory(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def _get_gpu_peak_memory_mb(self):
        if self.device.type == 'cuda':
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        return None

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        self._reset_gpu_peak_memory()
        train_start_time = time.time()
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        start_epoch, elapsed_before = self._resume_from_snapshot(
            setting, path, model_optim, early_stopping,
            scaler if self.args.use_amp else None)
        train_start_time -= elapsed_before

        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # The test split is deliberately untouched during training.  Earlier revisions
            # printed test loss every epoch (while still selecting checkpoints only on
            # validation loss).  Removing that diagnostic closes a human-in-the-loop leakage
            # channel and keeps the test set for the single final evaluation below.
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            # Written after the learning-rate step, so the snapshot's optimizer
            # already carries the rate that epoch+1 will run with.  A kill
            # between here and the next save therefore costs exactly one epoch.
            self._save_snapshot(
                setting, path, epoch + 1, time.time() - train_start_time,
                model_optim, early_stopping, scaler if self.args.use_amp else None)

        train_elapsed = time.time() - train_start_time

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if self._recovery_enabled():
            recovery.discard_snapshot(recovery.snapshot_path(path))

        return self.model, train_elapsed

    def _recovery_enabled(self):
        return bool(getattr(self.args, 'enable_recovery', False))

    def _resume_from_snapshot(self, setting, path, model_optim, early_stopping, amp_scaler):
        """Continue the run this setting already started, if one is on disk."""
        if not self._recovery_enabled():
            return 0, 0.0
        snapshot = recovery.read_snapshot(recovery.snapshot_path(path), setting)
        if snapshot is None:
            return 0, 0.0
        self.model.load_state_dict(snapshot['model'])
        model_optim.load_state_dict(snapshot['optimizer'])
        early_stopping.load_state_dict(snapshot['early_stopping'])
        if amp_scaler is not None:
            amp_scaler.load_state_dict(snapshot['amp_scaler'])
        recovery.restore_rng(snapshot['rng'])
        print('>>>>>>>resuming : {} at epoch {}>>>>>>>>>>>>>>>>'.format(
            setting, snapshot['next_epoch'] + 1))
        return snapshot['next_epoch'], snapshot['elapsed_seconds']

    def _save_snapshot(self, setting, path, next_epoch, elapsed_seconds,
                       model_optim, early_stopping, amp_scaler):
        if not self._recovery_enabled():
            return
        recovery.write_snapshot(recovery.snapshot_path(path), {
            'snapshot_version': recovery.SNAPSHOT_VERSION,
            'identity': setting,
            'next_epoch': next_epoch,
            'elapsed_seconds': elapsed_seconds,
            'model': self.model.state_dict(),
            'optimizer': model_optim.state_dict(),
            'early_stopping': early_stopping.state_dict(),
            'amp_scaler': amp_scaler.state_dict() if amp_scaler is not None else None,
            'rng': recovery.capture_rng(),
        })

    def test(self, setting, test=0, train_time=None):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            # standalone test (no training in this process): measure inference-only peak memory
            self._reset_gpu_peak_memory()

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        test_start_time = time.time()
        inference_time_total = 0.0
        total_samples = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                infer_start = time.time()
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                inference_time_total += time.time() - infer_start
                total_samples += batch_x.size(0)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        test_elapsed = time.time() - test_start_time
        inference_speed_ms = (inference_time_total / total_samples) * 1000  # ms per sample
        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        train_time_str = ', train_time:{:.2f}s'.format(train_time) if train_time is not None else ''
        inference_str = ', inference_time:{:.4f}ms/sample ({} samples, {:.4f}s total)'.format(inference_speed_ms, total_samples, inference_time_total)

        total_params, trainable_params = self._count_parameters()
        gpu_mem_peak = self._get_gpu_peak_memory_mb()
        gpu_mem_val = '{:.2f}'.format(gpu_mem_peak) if gpu_mem_peak is not None else 'nan'
        model_stats_str = ', params:{}, trainable_params:{}, gpu_mem_peak_mb:{}'.format(
            total_params, trainable_params, gpu_mem_val)

        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, r2:{}, dtw:{}, time:{:.2f}s{}{}{}'.format(
            mse, mae, rmse, mape, mspe, r2, dtw, test_elapsed, train_time_str, inference_str, model_stats_str))
        result_log_dir = os.path.dirname(os.path.abspath(self.args.result_log))
        os.makedirs(result_log_dir, exist_ok=True)
        f = open(self.args.result_log, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, r2:{}, dtw:{}, time:{:.2f}s{}{}{}'.format(
            mse, mae, rmse, mape, mspe, r2, dtw, test_elapsed, train_time_str, inference_str, model_stats_str))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # Generate publication-quality figures from .npy results
        try:
            from utils.visualization import generate_figures
            test_results_path = './test_results/' + setting + '/'
            generate_figures(input_path=folder_path, output_path=test_results_path)
        except Exception as e:
            print(f'Warning: figure generation failed: {e}')

        return
