{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6493ec58",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "import tensorflow.keras as keras\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.utils.multiclass import unique_labels\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f2130abd",
   "metadata": {},
   "outputs": [],
   "source": [
    "DATASET_PATH = \"data.json\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cc7a1a0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_data(dataset_path):\n",
    "    with open(dataset_path, \"r\") as fp:\n",
    "        data = json.load(fp)\n",
    "        \n",
    "        inputs = np.array(data[\"mfcc\"])\n",
    "        targets = np.array(data[\"labels\"])\n",
    "        \n",
    "        return inputs, targets\n",
    "    \n",
    "inputs, targets = load_data(DATASET_PATH)\n",
    "\n",
    "\n",
    "inputs_train , inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c0f4087",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = keras.Sequential([\n",
    "    #input layer\n",
    "    keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),\n",
    "    \n",
    "   \n",
    "    \n",
    "    #1st hidden layer\n",
    "    keras.layers.Dense(512, activation=\"relu\"),\n",
    "\n",
    "    #2nd hidden layer\n",
    "    keras.layers.Dense(256, activation=\"relu\"),\n",
    "    \n",
    "    #3rd hidden layer\n",
    "    keras.layers.Dense(64, activation=\"relu\"),\n",
    "    \n",
    "    #output layer\n",
    "    keras.layers.Dense(2, activation=\"softmax\")\n",
    "])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8c31f44f",
   "metadata": {},
   "outputs": [],
   "source": [
    "optimizer = keras.optimizers.Adam(learning_rate=0.0001)\n",
    "model.compile(optimizer=optimizer, loss=\"sparse_categorical_crossentropy\", metrics=[\"accuracy\"])\n",
    "model.summary()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d651e826",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs=50, batch_size=32)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7eeff699",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('tipe.model')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b45f4a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "label_encoder = LabelEncoder()\n",
    "encoded_targets_test = label_encoder.fit_transform(targets_test)\n",
    "\n",
    "# Compute the confusion matrix\n",
    "cm = confusion_matrix(encoded_targets_test, np.argmax(model.predict(inputs_test), axis=1), labels=[0, 1])\n",
    "\n",
    "# Define the class labels for plotting\n",
    "classes = label_encoder.classes_\n",
    "\n",
    "# Define a function to plot the confusion matrix\n",
    "def plot_confusion_matrix(cm, classes, title='Matrice de confusion'):\n",
    "    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks = np.arange(len(classes))\n",
    "    plt.xticks(tick_marks, classes, rotation=45)\n",
    "    plt.yticks(tick_marks, classes)\n",
    "    \n",
    "    fmt = 'd'\n",
    "    thresh = cm.max() / 2.\n",
    "    for i in range(cm.shape[0]):\n",
    "        for j in range(cm.shape[1]):\n",
    "            plt.text(j, i, format(cm[i, j], fmt),\n",
    "                     horizontalalignment=\"center\",\n",
    "                     color=\"white\" if cm[i, j] > thresh else \"black\")\n",
    "    \n",
    "    plt.ylabel('Données Réelles')\n",
    "    plt.xlabel('Données prédites')\n",
    "    plt.tight_layout()\n",
    "\n",
    "# Plot the confusion matrix\n",
    "plot_confusion_matrix(cm, classes)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b6f48559",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
