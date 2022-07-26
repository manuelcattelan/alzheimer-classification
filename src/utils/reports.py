def print_classification_report(classification_report):
    print(classification_report)
    for run in classification_report:
        print("Run n.{}:".format(run))
        print("\t>>> Accuracy:"
              "\n\t\t>>> mean: {:.1f}%"
              "\n\t\t>>> stdev: {:.1f}%"
              .format(classification_report[run][0][0][0],
                      classification_report[run][0][0][1]))
        print("\t>>> Precision:"
              "\n\t\t>>> mean: {:.1f}%"
              "\n\t\t>>> stdev: {:.1f}%"
              .format(classification_report[run][0][1][0],
                      classification_report[run][0][1][1]))
        print("\t>>> Recall:"
              "\n\t\t>>> mean: {:.1f}%"
              "\n\t\t>>> stdev: {:.1f}%"
              .format(classification_report[run][0][2][0],
                      classification_report[run][0][2][1]))
        print("\t>>> Run times:"
              "\n\t\t>>> Total training time: {:.3f}s"
              "\n\t\t>>> Total testing time: {:.3f}s"
              .format(classification_report[run][1][0],
                      classification_report[run][1][1]))
