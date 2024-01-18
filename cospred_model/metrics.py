"""Metrics to assess performance on peaks identification
Accuracy, Balanced Accuracy, Sensitivity, Specificity,
Precision, Recall, F1 score, top peak,
1 out of top 3 peaks, 2 out of top 3 peaks, 3 out of top 3 peaks
1-5 out of 5 top peaks,
Spectral distance, Masked spectral distance, pearson correlation, cosine similarity
"""
import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from collections import Counter

# from torchmetrics.regression import PearsonCorrCoef
# from scipy.stats import pearsonr
# from prosit_model import sanitize


def binarize(y, threshold):
    """
    removes peaks less than a threshold
    """
    y_binary = y.clone()
    y_binary[y_binary < threshold] = 0.0
    y_binary[y_binary >= threshold] = 1.0
    return y_binary


def masked_spectral_distance(pred, true):
    # Note, fragment ions that cannot exists (i.e. y20 for a 7mer) must have the value  -1.
    epsilon = torch.finfo(torch.float32).eps
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    pred_norm = torch.linalg.norm(true_masked)
    true_norm = torch.linalg.norm(pred_masked)
    pred_normalize = pred / pred_norm
    true_normalize = true / true_norm
    product = torch.sum(pred_normalize * true_normalize)
    arccos = torch.acos(product)
    return 2 * arccos / np.pi


def spectral_distance(pred, true):
    # # No need to mask, since there is no impossible -1 place holder for full prediction
    # epsilon = torch.finfo(torch.float32).eps
    # pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    # true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    pred_norm = torch.linalg.norm(pred)
    true_norm = torch.linalg.norm(true)
    true_normalize = true / true_norm
    pred_normalize = pred / pred_norm
    product = torch.sum(pred_normalize * true_normalize)
    arccos = torch.acos(product)
    return 2 * arccos / np.pi


def pearson_corr(pred, true):
    pred_norm = torch.linalg.norm(pred)
    true_norm = torch.linalg.norm(true)
    true_normalize = true / true_norm
    pred_normalize = pred / pred_norm
    corr = torch.sum((pred_normalize * true_normalize), dim=-1) / torch.sqrt(
        torch.sum((pred_normalize ** 2), dim=-1) * torch.sum((true_normalize ** 2), dim=-1))
    return corr


def cos_sim(pred, true):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(pred, true)


def roc_auc(pred_binary_arr, true_binary_arr):
    roc_auc = roc_auc_score(true_binary_arr, pred_binary_arr,
                            multi_class="ovr", average="micro")
    # summarize scores
    # print('Micro-averaged One-vs-Rest ROC AUC score=%.3f' % (roc_auc))
    return roc_auc


class ComputeMetrics:
    def __init__(self, true, pred, seq=None, charge=None, ce=None,
                 binarize_threshold=0.02,
                 eps=10 ** (-9),
                 ):

        self.seq = seq
        self.charge = charge
        self.ce = ce
        true_binary = binarize(true, threshold=binarize_threshold)
        pred_binary = binarize(pred, threshold=binarize_threshold)
        # vx = true - torch.mean(true, dim=0)
        # vy = pred - torch.mean(pred, dim=0)

        # # dimension demo
        # a = np.array([[0,0,0],[0,1,0],[1,0,1],[1,1,1]])
        # b = np.array([[0,0,1],[1,1,0],[1,1,1],[1,1,1]])
        # c = torch.from_numpy(a)
        # d = torch.from_numpy(b)
        # tp = torch.sum((c==1) & (d==1), dim = 1).float().mean()
        # fp = torch.sum((c==0) & (d==1), dim = 1).float().mean()
        # tn = torch.sum((c==0) & (d==0), dim = 1).float().mean()
        # fn = torch.sum((c==1) & (d==0), dim = 1).float().mean()
        # precision = tp/(tp+fp)

        # mass prediction focused metrics
        # self._tp = torch.sum((pred_binary == 1.0) & (true_binary == 1.0), axis=1).float()  # True positive
        # self._tn = torch.sum((pred_binary == 0.0) & (true_binary == 0.0), axis=1).float()  # True negative
        # self._fp = torch.sum((pred_binary == 1.0) & (true_binary == 0.0), axis=1).float()  # False positive
        # self._fn = torch.sum((pred_binary == 0.0) & (true_binary == 1.0), axis=1).float()  # False negative
        self._tp = torch.sum((pred_binary == 1.0) & (
            true_binary == 1.0), dim=1).float()  # True positive
        self._tn = torch.sum((pred_binary == 0.0) & (
            true_binary == 0.0), dim=1).float()  # True negative
        self._fp = torch.sum((pred_binary == 1.0) & (
            true_binary == 0.0), dim=1).float()  # False positive
        self._fn = torch.sum((pred_binary == 0.0) & (
            true_binary == 1.0), dim=1).float()  # False negative

        self.accuracy = (self._tp + self._tn) / \
            (self._tp + self._tn + self._fp + self._fn)
        self.sensitivity = self._tp / (self._tp + self._fn + eps)
        self.specificity = self._tn / (self._tn + self._fp + eps)
        self.balanced_accuracy = (self.sensitivity + self.specificity) / 2.0
        self.precision = self._tp / (self._tp + self._fp + eps)
        self.recall = self._tp / (self._tp + self._fn + eps)
        self.F1 = 2 * self.precision * self.recall / \
            (self.precision + self.recall + eps)

        # roc curve and auc on an imbalanced dataset
        self.true_binary = true_binary
        self.pred_binary = pred_binary
        # self.true_binary = true_binary.detach().cpu().numpy()
        # self.pred_binary = pred_binary.detach().cpu().numpy()

        # precision, recall, f1 score
        # self.precision_score, self.recall_score, self.f1_score, _ = precision_recall_fscore_support(y_true=self.true_binary, y_pred=self.pred_binary, average='macro', zero_division=1)
        # calculate ROC AUC scores
        # self.roc_auc = roc_auc(self.pred_binary, self.true_binary)

        # calculate roc curves
        # self.fpr, self.tpr, _ = roc_curve(self.true_binary.ravel(), self.pred_binary.ravel())

        # intensity focused metrics
        self.spectral_distance = spectral_distance(pred, true)
        self.masked_spectral_distance = masked_spectral_distance(pred, true)
        self.pcc = pearson_corr(pred, true)
        self.cos_sim = cos_sim(pred, true)

        batch_size = pred_binary.size(0)

        top1_mass = true.sort()[1][:, -1]
        top2_mass = true.sort()[1][:, -2]
        top3_mass = true.sort()[1][:, -3]
        top4_mass = true.sort()[1][:, -4]
        top5_mass = true.sort()[1][:, -5]

        self.top1_out_1 = pred_binary[torch.arange(batch_size), top1_mass]

        count_peaks = pred_binary[torch.arange(batch_size), top1_mass] + \
            pred_binary[torch.arange(batch_size), top2_mass] + \
            pred_binary[torch.arange(batch_size), top3_mass]

        self.top1_out_3 = binarize(count_peaks, threshold=1.0)
        self.top2_out_3 = binarize(count_peaks, threshold=2.0)
        self.top3_out_3 = binarize(count_peaks, threshold=3.0)

        count_peaks = pred_binary[torch.arange(batch_size), top1_mass] + \
            pred_binary[torch.arange(batch_size), top2_mass] + \
            pred_binary[torch.arange(batch_size), top3_mass] + \
            pred_binary[torch.arange(batch_size), top4_mass] + \
            pred_binary[torch.arange(batch_size), top5_mass]

        self.top1_out_5 = binarize(count_peaks, threshold=1.0)
        self.top2_out_5 = binarize(count_peaks, threshold=2.0)
        self.top3_out_5 = binarize(count_peaks, threshold=3.0)
        self.top4_out_5 = binarize(count_peaks, threshold=4.0)
        self.top5_out_5 = binarize(count_peaks, threshold=5.0)

    def return_metrics(self):
        metrics_dict = {'balanced_accuracy': self.balanced_accuracy,
                        'true_positive': self._tp,
                        'true_negative': self._tn,
                        'false_positive': self._fp,
                        'false_negative': self._fn,
                        'precision': self.precision,
                        'recall': self.recall,
                        'sensitivity': self.sensitivity,
                        'specificity': self.specificity,
                        'F1': self.F1,
                        'top1_out_1': self.top1_out_1,
                        'top1_out_3': self.top1_out_3,
                        'top2_out_3': self.top2_out_3,
                        'top3_out_3': self.top3_out_3,
                        'top1_out_5': self.top1_out_5,
                        'top2_out_5': self.top2_out_5,
                        'top3_out_5': self.top3_out_5,
                        'top4_out_5': self.top4_out_5,
                        'top5_out_5': self.top5_out_5,
                        'spectral_distance': self.spectral_distance,
                        'masked_spectral_distance': self.masked_spectral_distance,
                        'pearson_correlation': self.pcc,
                        'cosine_similarity': self.cos_sim,
                        # 'fpr': self.fpr,
                        # 'tpr': self.tpr,
                        # 'roc_auc':self.roc_auc,
                        # 'recall_score': self.recall_score,
                        # 'precision_score': self.precision_score,
                        # 'f1_score': self.f1_score,
                        }
        # for key, value in metrics_dict.items():
        #     try:
        #         print(f"Length of value for key '{key}': {len(value)}")
        #     except TypeError:
        #         print(f"Value for key '{key}' is not iterable.")
        return metrics_dict

    def return_metrics_byrecord(self):
        metrics_dict_byrecord = {'seq': self.seq,
                                 'charge': self.charge,
                                 'ce': self.ce,
                                 'balanced_accuracy': self.balanced_accuracy,
                                 'true_positive': self._tp,
                                 'true_negative': self._tn,
                                 'false_positive': self._fp,
                                 'false_negative': self._fn,
                                 'precision': self.precision,
                                 'recall': self.recall,
                                 'sensitivity': self.sensitivity,
                                 'specificity': self.specificity,
                                 'F1': self.F1,
                                 'top1_out_1': self.top1_out_1,
                                 'top1_out_3': self.top1_out_3,
                                 'top2_out_3': self.top2_out_3,
                                 'top3_out_3': self.top3_out_3,
                                 'top1_out_5': self.top1_out_5,
                                 'top2_out_5': self.top2_out_5,
                                 'top3_out_5': self.top3_out_5,
                                 'top4_out_5': self.top4_out_5,
                                 'top5_out_5': self.top5_out_5,
                                 'pearson_correlation': self.pcc,
                                 'cosine_similarity': self.cos_sim,
                                 }
        return metrics_dict_byrecord

    def return_metrics_mean(self):
        metrics = self.return_metrics()
        for key, value in metrics.items():
            metrics[key] = value.mean()
        return metrics

    def return_metrics_max(self):
        metrics = self.return_metrics()
        for key, value in metrics.items():
            metrics[key] = value.max()
        return metrics

    def plot_PRcurve_micro(self, plot_dir):
        metrics = self.return_metrics()

        # toggle to remove mass bins without peaks
        flag_removezero = False
        if flag_removezero is True:
            # OPTION 1: remove mass bin without peaks
            true_binary_colsum = np.sum(self.true_binary, axis=0)
            # pred_binary_colsum = np.sum(self.pred_binary, axis=0)
            colsum0_idx = np.where(true_binary_colsum != 0)[0]
        else:
            # OPTION 2: keep all mass bins
            colsum0_idx = range(self.true_binary.shape[1])

        # true_binary_nonzero = np.delete(self.true_binary[:, colsum0_idx], -1, axis=1) # remove the last column, which is the aggregate mass column
        # pred_binary_nonzero = np.delete(self.pred_binary[:, colsum0_idx], -1, axis=1)
        true_binary_nonzero = self.true_binary[:, colsum0_idx]
        pred_binary_nonzero = self.pred_binary[:, colsum0_idx]

        # precision-recall for each class
        precision = dict()
        recall = dict()
        average_precision = dict()

        # for i in range(true_binary_nonzero.shape[1]):
        #     precision[i], recall[i], _ = precision_recall_curve(true_binary_nonzero[:, i], pred_binary_nonzero[:, i])
        #     average_precision[i] = average_precision_score(true_binary_nonzero[:, i], pred_binary_nonzero[:, i])
        #     print('{}, Average precision-recall score: {}'.format(i, average_precision[i]))

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            true_binary_nonzero.ravel(), pred_binary_nonzero.ravel()
        )
        average_precision["micro"] = average_precision_score(
            true_binary_nonzero, pred_binary_nonzero, average="micro")

        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
            prevalence_pos_label=Counter(true_binary_nonzero.ravel())[
                1] / true_binary_nonzero.size,
        )
        display.plot(plot_chance_level=True)
        _ = display.ax_.set_title(
            "Micro-averaged Precision-Recall Charateristic")
        plt.savefig(plot_dir + 'PR_curve_micro.png')
        plt.close()
        return metrics

    def plot_PRcurve_sample(self, plot_dir):
        metrics = self.return_metrics()

        true_binary = self.true_binary
        pred_binary = self.pred_binary

        # precision-recall for each class
        precision = dict()
        recall = dict()
        # average_precision = dict()

        # A "sample-average": quantifying score on each sample, then averaging
        precision_score = [0]*true_binary.shape[0]
        recall_score = [0]*true_binary.shape[0]
        f1_score = [0]*true_binary.shape[0]
        for i in range(true_binary.shape[0]):
            precision[i], recall[i], _ = precision_recall_curve(
                true_binary[i, :], pred_binary[i, :])
            precision_score[i], recall_score[i], f1_score[i], _ = precision_recall_fscore_support(
                y_true=true_binary[i, :], y_pred=pred_binary[i, :], average='macro', zero_division=1)
            # average_precision[i] = average_precision_score(true_binary[i,:], pred_binary[i,:])
            # if i % 100 == 0:
            #     print('i = {}, precision= {}, recall = {}, precision_score={}, recall_score={}, f1_score={}, average_precision={}'.format(i, precision[i], recall[i], precision_score[i], recall_score[i], f1_score[i], average_precision[i]))

        # Assuming precision_score and recall_score are numpy arrays
        precision_score_arr = np.array(precision_score)
        recall_score_arr = np.array(recall_score)
        # Get the indices that would sort recall_score in ascending order
        sort_indices = np.argsort(recall_score_arr)
        # Use these indices to sort precision_score
        sorted_precision_score = precision_score_arr[sort_indices]
        sorted_recall_score = recall_score_arr[sort_indices]

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.scatter(
            sorted_recall_score,
            sorted_precision_score,
            label=f"All sample Precision-Recall",
            alpha=0.2,
            color="navy",
            linestyle="solid",
            linewidth=2,
        )
        # Draw a diagonal line
        plt.plot(
            [0, 1],
            [1, 0],
            label="random guess",
            color="darkorange",
            linestyle='-',
        )
        # Change the font size of the axis labels
        ax.tick_params(axis='both', which='major', labelsize=10)

        plt.axis("square")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("All Sample Precision Recall Characteristic", fontsize=14)
        plt.legend()
        plt.savefig(plot_dir + 'PR_sample_all.png')
        plt.close()

        # # Set the seed
        # np.random.seed(42)
        # # step PR plot for each class
        # plt.figure()
        # plt.axis("square")
        # # for i in np.random.choice(true_binary.shape[0] - 1, 10, replace=False):
        # for i in range(10):
        #     plt.step(recall[i], precision[i], where='post')
        # plt.xlabel('Recall', fontsize=12)
        # plt.ylabel('Precision', fontsize=12)
        # plt.title('Per Sample Precision-Recall Curve', fontsize=14)
        # # plt.legend()
        # plt.savefig(plot_dir + 'PR_curve_sample.png')
        # plt.close()
        return metrics

    def plot_PRcurve_macro(self, plot_dir):
        metrics = self.return_metrics()

        true_binary = self.true_binary
        pred_binary = self.pred_binary

        # precision-recall for each class
        precision = dict()
        recall = dict()
        average_precision = dict()

        # A "macro-average": quantifying score on each class, then averaging
        # precision_score = [0]*true_binary.shape[1]
        # recall_score = [0]*true_binary.shape[1]
        # f1_score = [0]*true_binary.shape[1]
        for i in range(true_binary.shape[1]):
            if np.any(true_binary[:, i] > 0):
                precision[i], recall[i], _ = precision_recall_curve(
                    true_binary[:, i], pred_binary[:, i])
                # precision_score[i], recall_score[i], f1_score[i], _ = precision_recall_fscore_support(y_true=true_binary[:,i], y_pred=pred_binary[:,i], average='macro', zero_division=1)
                average_precision[i] = average_precision_score(
                    true_binary[:, i], pred_binary[:, i])
            else:
                precision[i] = np.array([1])
                recall[i] = np.array([1])
                # precision_score[i] = 1
                # recall_score[i] = 1
                # f1_score[i] = 1
                average_precision[i] = 1
            # if i % 100 == 0:
            #     print('i = {}, precision= {}, recall = {}, precision_score={}, recall_score={}, f1_score={}, average_precision={}'.format(i, precision[i], recall[i], precision_score[i], recall_score[i], f1_score[i], average_precision[i]))
        average_precision["macro"] = np.mean(list(average_precision.values()))

        # Set the seed
        np.random.seed(42)
        # step PR plot for each class
        plt.figure()
        plt.axis("square")
        for i in np.random.choice(true_binary.shape[1] - 1, 10, replace=False):
            plt.step(recall[i], precision[i], where='post')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Macro-averaged Precision-Recall Charateristic', fontsize=14)
        # plt.legend()
        plt.savefig(plot_dir + 'PR_curve_macro.png')
        plt.close()
        return metrics

    def plot_ROCcurve_micro(self, plot_dir):
        metrics = self.return_metrics()

        RocCurveDisplay.from_predictions(
            self.true_binary.ravel(),
            self.pred_binary.ravel(),
            # self.true_binary,
            # self.pred_binary,
            name="micro-average OvR",
            color="darkorange",
            plot_chance_level=True,
        )
        plt.axis("square")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Micro-Averaged Receiver Operating Characteristic", fontsize=14)
        # plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        plt.legend()
        plt.savefig(plot_dir + 'ROC_curve_micro.png')
        plt.close()

        return metrics

    def plot_ROCcurve_macro(self, plot_dir):
        metrics = self.return_metrics()

        # y_true = np.array([[1,0,0],[0,1,0],[1,0,1],[1,1,1],[1,1,0]]) # 5x3
        # y_scores = np.array([[0,0,1],[1,1,0],[1,1,1],[1,1,1],[0,0,1]]) # 5x3
        y_true = self.true_binary
        y_scores = self.pred_binary

        # macro average ROC curve, compute TPR, FPR per instance across all massbins, then each instance is weighted equally
        auclist = dict()
        fpr = dict()
        tpr = dict()

        for i in range(y_true.shape[0]):
            if np.any(y_true[i, :] > 0):
                fpr[i], tpr[i], _ = roc_curve(y_true[i, :], y_scores[i, :])
                auclist[i] = auc(tpr[i], fpr[i])
                if i % 100 == 0:
                    print('i = {}, tpr= {}, fpr = {}'.format(
                        i, tpr[i], fpr[i]))
            else:
                fpr[i] = np.array([0])
                tpr[i] = np.array([1])
                auclist[i] = 1

        fpr_grid = np.linspace(0.0, 1.0, 1000)
        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(y_true.shape[0]):
            # linear interpolation
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
        # Average it and compute AUC
        mean_tpr /= y_true.shape[0]
        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        auclist["macro"] = auc(fpr["macro"], tpr["macro"])
        print(
            f"Macro-averaged One-vs-Rest ROC AUC score={auclist['macro']:.3f}")

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"macro-average ROC curve (AUC = {auclist['macro']:.3f})",
            color="navy",
            linestyle="solid",
            linewidth=4,
        )
        # Draw a diagonal line
        plt.plot(
            [0, 1],
            [0, 1],
            label="random guess (AUC = 0.5)",
            color="darkorange",
            linestyle='-',
        )
        # Change the font size of the axis labels
        ax.tick_params(axis='both', which='major', labelsize=10)

        plt.axis("square")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Macro-Averaged Receiver Operating Characteristic", fontsize=14)
        plt.legend()
        plt.savefig(plot_dir + 'ROC_curve_macro.png')
        plt.close()

        return metrics


class ComputeMetrics_CPU(ComputeMetrics):
    def __init__(self, true, pred, seq=None, charge=None, ce=None,
                 binarize_threshold=0.02,
                 eps=10 ** (-9),
                 ):

        # # OPTION 1: keep pred which is not NaN
        # print(pred.shape)
        # nan_row_mask = torch.any(torch.isnan(pred), dim=1)
        # nan_row_indices = torch.where(nan_row_mask)
        # nan_row_indices = nan_row_indices[0]

        # mask = torch.ones(pred.shape[0], dtype=bool)
        # mask[nan_row_indices] = False
        # pred = pred[mask]
        # true = true[mask]
        # print(pred.shape)
        # self.seq = seq[mask.tolist()]
        # self.charge = charge[mask.tolist()]
        # self.ce = ce[mask.tolist()]

        # OPTION 2: replace NaN to -1 mask for prediction
        pred[torch.isnan(pred)] = -1
        self.seq = seq
        self.charge = charge
        self.ce = ce
        true_binary = binarize(true, threshold=binarize_threshold)
        pred_binary = binarize(pred, threshold=binarize_threshold)
        # vx = true - torch.mean(true, dim=0)
        # vy = pred - torch.mean(pred, dim=0)

        self._tp = torch.sum((pred_binary == 1.0) & (
            true_binary == 1.0), dim=1).float()  # True positive
        self._tn = torch.sum((pred_binary == 0.0) & (
            true_binary == 0.0), dim=1).float()  # True negative
        self._fp = torch.sum((pred_binary == 1.0) & (
            true_binary == 0.0), dim=1).float()  # False positive
        self._fn = torch.sum((pred_binary == 0.0) & (
            true_binary == 1.0), dim=1).float()  # False negative

        self.accuracy = (self._tp + self._tn) / \
            (self._tp + self._tn + self._fp + self._fn)
        self.sensitivity = self._tp / (self._tp + self._fn + eps)
        self.specificity = self._tn / (self._tn + self._fp + eps)
        self.balanced_accuracy = (self.sensitivity + self.specificity) / 2.0
        self.precision = self._tp / (self._tp + self._fp + eps)
        self.recall = self._tp / (self._tp + self._fn + eps)
        self.F1 = 2 * self.precision * self.recall / \
            (self.precision + self.recall + eps)

        # roc curve and auc on an imbalanced dataset
        self.true_binary = true_binary.cpu().detach().numpy()
        self.pred_binary = pred_binary.cpu().detach().numpy()

        # precision, recall, f1 score
        self.precision_score, self.recall_score, self.f1_score, _ = precision_recall_fscore_support(
            y_true=self.true_binary, y_pred=self.pred_binary, average='macro', zero_division=1)
        # calculate ROC AUC scores
        self.roc_auc = roc_auc(self.pred_binary, self.true_binary)

        # calculate roc curves
        # self.ns_fpr, self.ns_tpr, _ = roc_curve(true_binary, pred_binary)
        self.fpr, self.tpr, _ = roc_curve(
            self.true_binary.ravel(), self.pred_binary.ravel())

        # intensity focused metrics
        self.spectral_distance = spectral_distance(pred, true)
        self.masked_spectral_distance = masked_spectral_distance(pred, true)
        self.pcc = pearson_corr(pred, true)
        self.cos_sim = cos_sim(pred, true)

        batch_size = pred_binary.size(0)

        top1_mass = true.sort()[1][:, -1]
        top2_mass = true.sort()[1][:, -2]
        top3_mass = true.sort()[1][:, -3]
        top4_mass = true.sort()[1][:, -4]
        top5_mass = true.sort()[1][:, -5]

        self.top1_out_1 = pred_binary[torch.arange(batch_size), top1_mass]

        count_peaks = pred_binary[torch.arange(batch_size), top1_mass] + \
            pred_binary[torch.arange(batch_size), top2_mass] + \
            pred_binary[torch.arange(batch_size), top3_mass]

        self.top1_out_3 = binarize(count_peaks, threshold=1.0)
        self.top2_out_3 = binarize(count_peaks, threshold=2.0)
        self.top3_out_3 = binarize(count_peaks, threshold=3.0)

        count_peaks = pred_binary[torch.arange(batch_size), top1_mass] + \
            pred_binary[torch.arange(batch_size), top2_mass] + \
            pred_binary[torch.arange(batch_size), top3_mass] + \
            pred_binary[torch.arange(batch_size), top4_mass] + \
            pred_binary[torch.arange(batch_size), top5_mass]

        self.top1_out_5 = binarize(count_peaks, threshold=1.0)
        self.top2_out_5 = binarize(count_peaks, threshold=2.0)
        self.top3_out_5 = binarize(count_peaks, threshold=3.0)
        self.top4_out_5 = binarize(count_peaks, threshold=4.0)
        self.top5_out_5 = binarize(count_peaks, threshold=5.0)
