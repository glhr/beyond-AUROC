import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from ood_metrics import calc_standard_metrics, calc_autc


def plot_ood_scores(id_data,ood_data,model_name=None,plot=True,save=False,clip=True,plot_labels=True,plot_threshs=True,return_metrics=False,plot_tight=True):

    if clip:
        ood_data = ood_data[ood_data<=1]
        ood_data = ood_data[ood_data>=0]
        id_data = id_data[id_data<=1]
        id_data = id_data[id_data>=0]
    
    c = dict()

    c["name"] = model_name if model_name is not None else ""
    c["preds"] = np.concatenate([id_data, ood_data])
    c["labels"] = np.concatenate([np.zeros_like(id_data),np.ones_like(ood_data)]).astype(int)
    
    c["metrics"] = calc_standard_metrics(c["preds"],c["labels"],pos_label=1)
    pprint(c["metrics"], width=1)
    
    if plot:
        count, bins, ignored = plt.hist([id_data,ood_data], 20, density=True, color=["cornflowerblue","violet"], label=["ID", "OOD"], alpha=0.7, histtype='step',fill=True)
        

        plt.xlim(left=0, right=1)
        plt.locator_params(axis="y", integer=True, tight=False)
        if plot_threshs:
            for thresh_color,thresh_label in zip(["purple","green"],["thresh_95tpr","thresh_95tnr"]):
                plt.axvline(x = c["metrics"][thresh_label], label = f"@{thresh_label.replace('thresh_','')}", color = thresh_color, linestyle = "--")
        if plot_labels:
            plt.xlabel("OOD scores")
            plt.ylabel("Density")

            handles, labels = plt.gca().get_legend_handles_labels()
            order = [1,0]
            plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", mode="expand", borderaxespad=0, ncol=4 if plot_threshs else 2)
        if plot_tight: plt.tight_layout(pad=0)
        if save: plt.savefig(f"ood_plots/histo-{model_name}-{c['metrics']['auroc']:.5f}.pdf")
        plt.show()
        
    autc_dict = calc_autc(c["preds"],c["labels"])

    print(f"auFNR {autc_dict['aufnr']:.4f}, auFPR {autc_dict['aufpr']:.4f}")
    print(f"--> AUTC {autc_dict['autc']:.4f}")
    
    if plot:
        plt.plot(autc_dict["sorted_thresh"], autc_dict["sorted_fnr"],alpha=1,color="orange")
        plt.plot(autc_dict["sorted_thresh"], autc_dict["sorted_fpr"],alpha=1,color="green")
        
        plt.fill_between(autc_dict["sorted_thresh"], 0, autc_dict["sorted_fpr"],alpha=0.5, color="green", label="FPR")
        plt.fill_between(autc_dict["sorted_thresh"], 0, autc_dict["sorted_fnr"],alpha=0.5, color="orange", label="FNR")
        plt.xlim(left=0, right=1)
        plt.yticks([0,1])
        
        if plot_labels:
            plt.xlabel("OOD detection threshold")
            plt.ylabel("Rate")
            plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", borderaxespad=0, ncol=2)
        if plot_tight: plt.tight_layout(pad=0)
        if save: plt.savefig(f"ood_plots/autc-{model_name}-{c['metrics']['auroc']:.5f}.pdf")
        plt.show()

    if return_metrics:
        return c