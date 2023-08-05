from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_targets_test = label_encoder.fit_transform(targets_test)

# Compute the confusion matrix
cm = confusion_matrix(encoded_targets_test, np.argmax(model.predict(inputs_test), axis=1), labels=[0, 1])

# Define the class labels for plotting
classes = label_encoder.classes_

# Define a function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, title='Matrice de confusion'):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Données Réelles')
    plt.xlabel('Données prédites')
    plt.tight_layout()

# Plot the confusion matrix
plot_confusion_matrix(cm, classes)
plt.show()
