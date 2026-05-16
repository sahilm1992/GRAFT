import logging
import torch
logger = logging.getLogger("main")


class Logger(object):
    def __init__(self, primary_metric_name: str = "accuracy", greater_is_better: bool = True):
        self.overall = []      # [(train, val, test)]
        self.per_class = []    # list of dicts per epoch
        self.metrics = []      # list of dicts per epoch
        self.primary_metric_name = primary_metric_name
        self.greater_is_better = greater_is_better

    def add_result(self, result):
        self.overall.append(result["overall"])
        self.per_class.append(result["per_class"])
        if "metrics" in result:
            self.metrics.append(result["metrics"])

    def summarize(self):
        overall = torch.tensor(self.overall)
        train_acc = overall[:, 0]
        val_acc   = overall[:, 1]
        test_acc  = overall[:, 2]

        if self.greater_is_better:
            best_epoch = val_acc.argmax().item()
        else:
            best_epoch = val_acc.argmin().item()

        summary = {
            # ---------- Overall primary metric ----------
            "best_epoch": best_epoch + 1,
            f"train_{self.primary_metric_name}_best_val": train_acc[best_epoch].item(),
            f"val_{self.primary_metric_name}_best": val_acc[best_epoch].item(),
            f"test_{self.primary_metric_name}_best": test_acc[best_epoch].item(),
            f"final_train_{self.primary_metric_name}": train_acc[-1].item(),
            f"final_val_{self.primary_metric_name}": val_acc[-1].item(),
            f"final_test_{self.primary_metric_name}": test_acc[-1].item(),
            f"mean_val_{self.primary_metric_name}": val_acc.mean().item(),
            f"std_val_{self.primary_metric_name}": val_acc.std().item(),
            f"train_val_{self.primary_metric_name}_gap": (train_acc[best_epoch] - val_acc[best_epoch]).item(),
            f"val_test_{self.primary_metric_name}_gap": (val_acc[best_epoch] - test_acc[best_epoch]).item(),

            # ---------- Per-class accuracy ----------
            "per_class_accuracy": {
                "train": self.per_class[best_epoch]["train"],
                "val": self.per_class[best_epoch]["val"],
                "test": self.per_class[best_epoch]["test"]
            }
        }

        if self.metrics:
            best_metrics = self.metrics[best_epoch]
            final_metrics = self.metrics[-1]
            
            for split in ["train", "val", "test"]:
                for metric_name, val in best_metrics[split].items():
                    # Avoid duplicating accuracy if already present
                    if metric_name == "acc":
                        continue
                    summary[f"{split}_{metric_name}_best_val"] = val
                
                for metric_name, val in final_metrics[split].items():
                    if metric_name == "acc":
                        continue
                    summary[f"final_{split}_{metric_name}"] = val

        return summary

    def print_statistics(self):
        s = self.summarize()
        val_key = f"val_{self.primary_metric_name}_best"
        test_key = f"test_{self.primary_metric_name}_best"
        if self.primary_metric_name == "accuracy":
            logger.info(f"Best Val Accuracy: {100 * s[val_key]:.2f}%")
            logger.info(f"Test Accuracy @ Best Val: {100 * s[test_key]:.2f}%")
        else:
            logger.info(f"Best Val {self.primary_metric_name.upper()}: {s[val_key]:.6f}")
            logger.info(f"Test {self.primary_metric_name.upper()} @ Best Val: {s[test_key]:.6f}")
