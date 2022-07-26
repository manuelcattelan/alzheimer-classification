import numpy as np


def print_classification_report(input, classification_report):
    accuracy_mean_list = []
    accuracy_variance_list = []
    precision_mean_list = []
    precision_variance_list = []
    recall_mean_list = []
    recall_variance_list = []
    print("CLASSIFICATION REPORT FOR '{}'". format(input))
    for run in classification_report:
        acc_mean = classification_report[run][0][0][0]
        acc_var = classification_report[run][0][0][1]
        prec_mean = classification_report[run][0][1][0]
        prec_var = classification_report[run][0][1][1]
        rec_mean = classification_report[run][0][2][0]
        rec_var = classification_report[run][0][2][1]
        train_time = classification_report[run][1][0]
        test_time = classification_report[run][1][1]
        print("Run [{}]:".format(run))
        print("\tAccuracy:"
              "\tmean={:.1f}%"
              "\tstdev={:.1f}%"
              .format(acc_mean, np.sqrt(acc_var)))
        print("\tPrecision:"
              "\tmean={:.1f}%"
              "\tstdev={:.1f}%"
              .format(prec_mean, np.sqrt(prec_var)))
        print("\tRecall:"
              "\t\tmean={:.1f}%"
              "\tstdev={:.1f}%"
              .format(rec_mean, np.sqrt(rec_var)))
        print("\tTimes:"
              "\t\ttraining={:.3f}s"
              "\ttesting={:.3f}s"
              .format(train_time, test_time))
        accuracy_mean_list.append(acc_mean)
        precision_mean_list.append(prec_mean)
        recall_mean_list.append(rec_mean)
        accuracy_variance_list.append(acc_var)
        precision_variance_list.append(prec_var)
        recall_variance_list.append(rec_var)
    print("")
    print("Model accuracy:"
          "\t\tmean={:.1f}%"
          "\tstdev={:.1f}%"
          .format(np.mean(accuracy_mean_list),
                  np.sqrt(np.mean(accuracy_variance_list))))
    print("Model precision:"
          "\tmean={:.1f}%"
          "\tstdev={:.1f}%"
          .format(np.mean(precision_mean_list),
                  np.sqrt(np.mean(precision_variance_list))))
    print("Model recall:"
          "\t\tmean={:.1f}%"
          "\tstdev={:.1f}%"
          .format(np.mean(recall_mean_list),
                  np.sqrt(np.mean(recall_variance_list))))
