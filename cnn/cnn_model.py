import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# --- KHOI INCEPTION (DA LUONG) ---
def inception_module(x, filters):
    # Luong 1: Nhin chi tiet (Kernel 1)
    path1 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
    
    # Luong 2: Nhin vua (Kernel 3)
    path2 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
    
    # Luong 3: Nhin xa (Kernel 5)
    path3 = layers.Conv1D(filters, 5, padding='same', activation='relu')(x)
    
    # Gop 3 luong
    return layers.Concatenate(axis=-1)([path1, path2, path3])

def tao_mo_hinh_cnn(input_shape, so_lop):
    inputs = layers.Input(shape=input_shape)
    
    # Layer khoi dau
    x = layers.Conv1D(32, 1, activation='relu')(inputs)
    
    # --- BLOCK 1 ---
    x = inception_module(x, 16)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.05)(x)  # GIẢM SÂU: Từ 0.2 -> 0.05 (Chỉ tắt 5%)
    
    # --- BLOCK 2 ---
    x = inception_module(x, 32)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.1)(x)   # GIẢM SÂU: Từ 0.3 -> 0.1
    
    # --- BLOCK 3 ---
    x = inception_module(x, 64)
    x = layers.BatchNormalization()(x) 
    x = layers.GlobalMaxPooling1D()(x)
    
    # --- OUTPUT ---
    x = layers.Dropout(0.2)(x)  
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.Dropout(0.2)(x)   
    
    outputs = layers.Dense(so_lop, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # Learning rate: 0.0005
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    # loss
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model