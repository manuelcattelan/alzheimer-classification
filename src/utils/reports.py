def export_clf_representation(clf, X, y, features, labels, output_path, index):
    output_dirname = Path(os.path.dirname(output_path))
    output_dirname.mkdir(parents=True, exist_ok=True)
    output_path_without_suffix = str(Path(output_path).with_suffix('')) + '_tree'
    output_path_with_suffix = Path(output_path_without_suffix).with_suffix('.png')

    # Train model on entire task data
    trained_clf = train_clf(clf, X, y, index)
    # map boolean labels to strings
    labels = [*map(({0: 'Sano', 1: 'Malato'}).get, labels)]

    # Plot trained model and export its .png representation
    tree.plot_tree(trained_clf, feature_names=features, class_names=labels, filled=True, rounded=True)
    plt.savefig(output_path_with_suffix, bbox_inches="tight", dpi=1200)
    plt.close()

def export_clf_performance(cm, performance, labels, output_path):
    output_dirname = Path(os.path.dirname(output_path))
    output_dirname.mkdir(parents=True, exist_ok=True)
    output_path_without_suffix = str(Path(output_path).with_suffix('')) + '_cm'
    output_path_with_suffix = Path(output_path_without_suffix).with_suffix('.png')

    # map boolean labels to strings
    labels = [*map(({0: 'Sano', 1: 'Malato'}).get, labels)]
    # build performance text to display under cm heatmap
    performance_text = '\n\nAccuracy: {:.1f}%\nPrecision: {:.1f}%\nRecall: {:.1f}%\nF1 score: {:.1f}%'.format(performance[0],
                                                                                                              performance[1],
                                                                                                              performance[2],
                                                                                                              performance[3])

    # Build heatmap with confusion matrix
    ax = sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues')
    ax.set_xlabel('\nPredicted values' + performance_text)
    ax.set_ylabel('Actual values')

    # Export heatmap to output path
    plt.savefig(output_path_with_suffix, bbox_inches="tight", dpi=400) 
    plt.close()
