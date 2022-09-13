
// import * as np from 'numpy';
// import * as tf from 'tensorflow';
const tf = require("tensorflow")
// import { keras } from 'tensorflow/python';
const keras=require("tensorflow/python/keras")
import { layers } from 'keras';
import { Sequential } from 'keras/models';
import * as pathlib from 'pathlib';
var AUTOTUNE, batch_size, class_names, data_augmentation, data_dir, epochs, first_image, history, image_batch, img, img_array, img_height, img_width, labels_batch, model, normalization_layer, normalized_ds, num_classes, predictions, score, train_ds, val_ds;
data_dir = "/content/products/";
data_dir = new pathlib.Path(data_dir);
batch_size = 2;
img_height = 180;
img_width = 180;
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, {
  "validation_split": 0.2,
  "subset": "training",
  "seed": 123,
  "image_size": [img_height, img_width],
  "batch_size": batch_size
});
val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, {
  "validation_split": 0.2,
  "subset": "validation",
  "seed": 123,
  "image_size": [img_height, img_width],
  "batch_size": batch_size
});
class_names = train_ds.class_names;
console.log(class_names);
console.log(val_ds.class_names);
AUTOTUNE = tf.data.AUTOTUNE;
train_ds = train_ds.cache().shuffle(1000).prefetch({
  "buffer_size": AUTOTUNE
});
val_ds = val_ds.cache().prefetch({
  "buffer_size": AUTOTUNE
});
normalization_layer = new layers.Rescaling(1.0 / 255);
normalized_ds = train_ds.map((x, y) => {
  return [normalization_layer(x), y];
});
[image_batch, labels_batch] = next(iter(normalized_ds));
first_image = image_batch[0];
console.log(np.min(first_image), np.max(first_image));
num_classes = class_names.length;
model = new Sequential([new layers.Rescaling(1.0 / 255, {
  "input_shape": [img_height, img_width, 3]
}), new layers.Conv2D(16, 3, {
  "padding": "same",
  "activation": "relu"
}), new layers.MaxPooling2D(), new layers.Conv2D(32, 3, {
  "padding": "same",
  "activation": "relu"
}), new layers.MaxPooling2D(), new layers.Conv2D(64, 3, {
  "padding": "same",
  "activation": "relu"
}), new layers.MaxPooling2D(), new layers.Flatten(), new layers.Dense(128, {
  "activation": "relu"
}), new layers.Dense(num_classes)]);
model.compile({
  "optimizer": "adam",
  "loss": new tf.keras.losses.SparseCategoricalCrossentropy({
    "from_logits": true
  }),
  "metrics": ["accuracy"]
});
model.summary();
epochs = 10;
history = model.fit(train_ds, {
  "validation_data": val_ds,
  "epochs": epochs
});
data_augmentation = new keras.Sequential([new layers.RandomFlip("horizontal", {
  "input_shape": [img_height, img_width, 3]
}), new layers.RandomRotation(0.3), new layers.RandomZoom(0.3)]);
model = new Sequential([data_augmentation, new layers.Rescaling(1.0 / 255), new layers.Conv2D(16, 3, {
  "padding": "same",
  "activation": "relu"
}), new layers.MaxPooling2D(), new layers.Conv2D(32, 3, {
  "padding": "same",
  "activation": "relu"
}), new layers.MaxPooling2D(), new layers.Conv2D(64, 3, {
  "padding": "same",
  "activation": "relu"
}), new layers.MaxPooling2D(), new layers.Dropout(0.3), new layers.Flatten(), new layers.Dense(128, {
  "activation": "relu"
}), new layers.Dense(num_classes, {
  "name": "outputs"
})]);
model.compile({
  "optimizer": "adam",
  "loss": new tf.keras.losses.SparseCategoricalCrossentropy({
    "from_logits": true
  }),
  "metrics": ["accuracy"]
});
model.build(image_batch.shape);
model.summary();
epochs = 15;
history = model.fit(train_ds, {
  "validation_data": val_ds,
  "epochs": epochs
});
data_dir = "./homepod-mini.jpg";
data_dir = new pathlib.Path(data_dir);
img = tf.keras.utils.load_img(data_dir, {
  "target_size": [img_height, img_width]
});
img_array = tf.keras.utils.img_to_array(img);
img_array = tf.expand_dims(img_array, 0);
predictions = model.predict(img_array);
score = tf.nn.softmax(predictions[0]);
console.log("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)));
