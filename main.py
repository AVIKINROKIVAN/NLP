import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import partial
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import random
import math
import time
import spacy
import warnings
import os
warnings.filterwarnings('ignore')

# Конфигурация
class Config:
    SEED = 1234
    # --- КЛЮЧЕВОЙ ПАРАМЕТР ДЛЯ ВРЕМЕНИ ---
    # Начните с ОЧЕНЬ малой доли, например 0.01 (1%) или 0.02 (2%)
    # Замерьте время 1 эпохи и увеличьте долю, пока время не станет ~50-55 минут
    DATA_FRACTION = 0.8 # Пример: используем только 2% данных!

    # Проверьте, помещается ли этот батч в память RTX 3070 с УПРОЩЕННОЙ моделью
    # Если нет ('CUDA out of memory'), уменьшите до 128 или 64
    BATCH_SIZE = 128

    ENC_EMB_DIM = 128   # Размерность эмбеддинга энкодера
    DEC_EMB_DIM = 128   # Размерность эмбеддинга декодера
    HID_DIM = 128       # Скрытая размерность GRU (для энкодера и декодера)
    ENC_DROPOUT = 0.3  # Dropout энкодера
    DEC_DROPOUT = 0.3  # Dropout декодера
    
    # # --- Ограничение Длины ---
    # MAX_SRC_LEN = 80   # Макс. длина последовательности источника (аннотации)
    # MAX_TRG_LEN = 25   # Макс. длина последовательности цели (заголовка)

    N_EPOCHS = 30       # Количество эпох обучения
    CLIP = 1           # Градиентный клиппинг
    TEACHER_FORCING_RATIO = 0.5 # Вероятность использования teacher forcing
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = 'ultra_fast_gru_generator.pt' # Имя файла для сохранения модели
    TRAIN_DATA = 'train.csv' # Путь к обучающим данным
    TEST_DATA = 'test.csv'   # Путь к тестовым данным
    MIN_FREQ = 5       # Минимальная частота слова для включения в словарь
    NUM_WORKERS = 8    # Количество процессов для загрузки данных (0 для CPU)

    # Индексы спец. токенов (определяются в TextProcessor)
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

# Инициализация
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
# torch.backends.cudnn.deterministic = True # Убрано для benchmark=True
torch.backends.cudnn.benchmark = True      # Включаем для потенциального ускорения

# Токенизатор
try:
    spacy_en = spacy.load('en_core_web_sm')
except OSError:
    print('Downloading language model for the spaCy tokenization\n'
          "(don't worry, this will only happen once)")
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    spacy_en = spacy.load('en_core_web_sm')

def tokenize(text):
    """Токенизирует текст с использованием spaCy."""
    return [tok.text for tok in spacy_en.tokenizer(str(text)) if not tok.text.isspace()]

# Обработчик текста
class TextProcessor:
    """Класс для создания словаря и нумерикализации текста."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.n_words = 0
        # Добавляем спец. токены в фиксированном порядке
        self.add_word('<pad>') # Config.PAD_IDX = 0
        self.add_word('<sos>') # Config.SOS_IDX = 1
        self.add_word('<eos>') # Config.EOS_IDX = 2
        self.add_word('<unk>') # Config.UNK_IDX = 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
            self.vocab.add(word)

    def build_vocab(self, texts, min_freq=1):
        """Строит словарь на основе списка текстов."""
        word_counts = {}
        print("Building vocabulary...")
        for text in tqdm(texts, desc="Counting words"):
            for word in tokenize(text):
                word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in tqdm(word_counts.items(), desc="Adding words to vocab"):
            if count >= min_freq:
                self.add_word(word)
        print(f"Vocabulary size: {self.n_words} words (min_freq={min_freq})")


    def numericalize(self, text):
        """Преобразует токенизированный текст в список индексов."""
        tokens = tokenize(text)
        return [self.word2idx.get(token, Config.UNK_IDX) for token in tokens]

# Датасет
class ArticleDataset(Dataset):
    def __init__(self, abstracts, titles, text_processor):
        self.abstracts = abstracts
        self.titles = titles
        self.text_processor = text_processor
        
    def __len__(self):
        return len(self.abstracts)  # Используем длину abstracts для определения размера датасета
    
    def __getitem__(self, idx):
        abstract = str(self.abstracts.iloc[idx]) if hasattr(self.abstracts, 'iloc') else str(self.abstracts[idx])
        title = str(self.titles.iloc[idx]) if hasattr(self.titles, 'iloc') else str(self.titles[idx])

        abstract_num = [Config.SOS_IDX] + self.text_processor.numericalize(abstract) + [Config.EOS_IDX]
        title_num = [Config.SOS_IDX] + self.text_processor.numericalize(title) + [Config.EOS_IDX]

        return {
            'src': torch.LongTensor(abstract_num),
            'trg': torch.LongTensor(title_num),
            'src_len': len(abstract_num)
        }

# Функция для подготовки батчей
def collate_fn(batch, pad_idx):
    """Собирает батч данных, применяя паддинг."""
    src_list = [item['src'] for item in batch]
    trg_list = [item['trg'] for item in batch]
    src_len_list = [item['src_len'] for item in batch]

    # Применяем паддинг до максимальной длины в батче
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, padding_value=pad_idx, batch_first=False)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_list, padding_value=pad_idx, batch_first=False)

    src_len_tensor = torch.LongTensor(src_len_list)

    return {
        'src': src_padded,
        'trg': trg_padded,
        'src_len': src_len_tensor
    }

# Модель Encoder (Однонаправленный GRU)
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=Config.PAD_IDX)
        self.rnn = nn.GRU(emb_dim, hid_dim) # Без bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src: [src_len, batch_size]
        # src_len: [batch_size]

        embedded = self.dropout(self.embedding(src))
        # embedded: [src_len, batch_size, emb_dim]

        # Нужно передать длины на CPU как list или numpy array
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_len.cpu().tolist(), # Используем tolist()
            enforce_sorted=False
        )

        # packed_outputs: packed sequence
        # hidden: [num_layers * num_directions, batch_size, hid_dim] -> [1, batch_size, hid_dim]
        packed_outputs, hidden = self.rnn(packed_embedded)

        # outputs: [src_len, batch_size, hid_dim]
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs содержит скрытые состояния для каждого токена
        # hidden содержит последнее скрытое состояние
        return outputs, hidden

# Механизм внимания
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        # Вход: [dec_hid_state, enc_output_state] -> [dec_hid_dim + enc_hid_dim]
        # Так как enc_hid_dim == dec_hid_dim == HID_DIM
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch size, dec_hid_dim] (from decoder, squeezed)
        # encoder_outputs: [src_len, batch_size, enc_hid_dim]
        # mask: [batch size, src_len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # Повторяем скрытое состояние декодера src_len раз
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # -> [batch size, src len, dec_hid_dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2) # -> [batch size, src len, enc_hid_dim]

        # Вычисляем энергию
        # attn_input shape: [batch size, src len, dec_hid_dim + enc_hid_dim]
        attn_input = torch.cat((hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(attn_input)) # -> [batch size, src len, dec_hid_dim]

        energy = energy.permute(0, 2, 1) # -> [batch size, dec_hid_dim, src len]

        # v: [dec_hid_dim] -> [batch size, 1, dec_hid_dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # attention scores: [batch size, 1, src_len] -> [batch size, src_len]
        attention = torch.bmm(v, energy).squeeze(1)

        # Применяем маску (чтобы не учитывать паддинг в источнике)
        attention = attention.masked_fill(mask == 0, -1e10) # Используем маску напрямую

        return F.softmax(attention, dim=1) # -> [batch size, src_len]

# Модель Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=Config.PAD_IDX)
        # Вход GRU: [embedding, context_vector] -> [emb_dim + enc_hid_dim]
        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim)
        # Вход FC: [gru_output, context_vector, embedding] -> [dec_hid_dim + enc_hid_dim + emb_dim]
        self.out = nn.Linear(dec_hid_dim + enc_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input: [batch size] (индекс предыдущего токена)
        # hidden: [1, batch size, dec_hid_dim] (скрытое состояние от энкодера/предыдущего шага)
        # encoder_outputs: [src len, batch size, enc_hid_dim]
        # mask: [batch size, src len]

        input = input.unsqueeze(0) # -> [1, batch size]

        embedded = self.dropout(self.embedding(input)) # -> [1, batch size, emb dim]

        # Получаем веса внимания
        # Передаем hidden[-1] (последний слой) в attention
        # hidden.squeeze(0) -> [batch size, dec_hid_dim]
        a = self.attention(hidden.squeeze(0), encoder_outputs, mask) # -> [batch size, src len]
        a = a.unsqueeze(1) # -> [batch size, 1, src len]

        # context vector (weighted sum of encoder_outputs)
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # -> [batch size, src len, enc_hid_dim]
        # weighted: [batch size, 1, enc_hid_dim]
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2) # -> [1, batch size, enc_hid_dim]

        # Вход для GRU
        rnn_input = torch.cat((embedded, weighted), dim=2) # -> [1, batch size, emb_dim + enc_hid_dim]

        # output: [1, batch size, dec_hid_dim]
        # hidden: [1, batch size, dec_hid_dim] (новое скрытое состояние)
        output, hidden = self.rnn(rnn_input, hidden) # hidden передается как [1, batch, hid]

        # Убираем размерность seq_len=1 для конкатенации
        embedded = embedded.squeeze(0) # -> [batch size, emb dim]
        output = output.squeeze(0)     # -> [batch size, dec hid dim]
        weighted = weighted.squeeze(0) # -> [batch size, enc hid dim]

        # Предсказание следующего токена
        # fc_input: [batch size, dec_hid_dim + enc_hid_dim + emb_dim]
        fc_input = torch.cat((output, weighted, embedded), dim=1)
        prediction = self.out(fc_input) # -> [batch size, output_dim]

        return prediction, hidden, a.squeeze(1) # Возвращаем веса внимания для возможной визуализации

# Полная модель Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, sos_idx, eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def create_mask(self, src):
        # src: [src_len, batch_size]
        mask = (src != self.pad_idx).permute(1, 0) # -> [batch size, src_len]
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        max_len = trg.shape[0] if trg is not None else 100  # Можно установить разумное максимальное значение
        
        outputs = torch.zeros(max_len, batch_size, self.decoder.output_dim).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_len)
        input = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=self.device) if trg is None else trg[0,:]
        mask = self.create_mask(src)

        for t in range(1, max_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force and trg is not None else top1
            if trg is None and (input == self.eos_idx).all():
                return outputs[:t+1]
        return outputs

# Функции обучения
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch in tqdm(iterator, desc='Training', leave=False):
        src = batch['src'].to(Config.DEVICE)
        trg = batch['trg'].to(Config.DEVICE)
        src_len = batch['src_len'] # Длины остаются на CPU для pack_padded_sequence

        optimizer.zero_grad()
        # Передаем trg для teacher forcing
        output = model(src, src_len, trg, Config.TEACHER_FORCING_RATIO)

        # output: [trg_len, batch_size, output_dim]
        # trg: [trg_len, batch_size]

        # Убираем <sos> токен и выравниваем для CrossEntropyLoss
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim) # -> [(trg_len-1)*batch_size, output_dim]
        trg = trg[1:].view(-1) # -> [(trg_len-1)*batch_size]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Функция оценки
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in tqdm(iterator, desc='Evaluating', leave=False):
            src = batch['src'].to(Config.DEVICE)
            trg = batch['trg'].to(Config.DEVICE)
            src_len = batch['src_len'] # Остается на CPU

            # Передаем trg, но teacher_forcing_ratio = 0
            output = model(src, src_len, trg, 0) # Teacher forcing выключен

            # Убираем <sos> токен и выравниваем
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Генерация заголовков
def generate_title(model, text_processor, abstract):
    model.eval()
    numericalized = [Config.SOS_IDX] + text_processor.numericalize(abstract) + [Config.EOS_IDX]
    src_len = torch.LongTensor([len(numericalized)]).to('cpu')
    src = torch.LongTensor(numericalized).unsqueeze(1).to(Config.DEVICE)

    with torch.no_grad():
        output = model(src, src_len, None, 0)

    output_indices = output.argmax(2).squeeze(1).cpu().tolist()
    title_tokens = []
    for idx in output_indices:
        if idx == Config.EOS_IDX:
            break
        if idx != Config.SOS_IDX and idx != Config.PAD_IDX:
             title_tokens.append(text_processor.idx2word.get(idx, '<unk>'))
    return ' '.join(title_tokens).replace('<unk>', '').strip()

def test_model(model, text_processor):
    """Функция для тестирования модели и генерации результатов"""
    print("\nLoading test data for generation...")
    try:
        test_df = pd.read_csv(Config.TEST_DATA)
        if 'abstract' not in test_df.columns:
            print(f"Error: Column 'abstract' not found in {Config.TEST_DATA}")
            return
        test_abstracts = test_df['abstract']
    except FileNotFoundError:
        print(f"Error: Test data file not found at {Config.TEST_DATA}")
        return

    model.eval()
    titles = []
    print(f"\nGenerating titles for {len(test_abstracts)} test abstracts...")
    
    for i in tqdm(range(len(test_abstracts)), desc='Generating titles'):
        abstract = test_abstracts.iloc[i]
        try:
            title = generate_title(model, text_processor, abstract)
            titles.append(title)
        except Exception as e:
            print(f"\nError generating title for abstract index {i}: {e}")
            titles.append("")

    if len(titles) == len(test_abstracts):
        submission_filename = 'submission_generated.csv'
        pd.DataFrame({'abstract': test_abstracts, 'title': titles}).to_csv(submission_filename, index=False)
        print(f"\nSubmission file '{submission_filename}' successfully created!")
    else:
        print(f"\nError: Generated {len(titles)} titles but expected {len(test_abstracts)}")

# Основная функция
def main():
    start_total_time = time.time()
    
    # Выбор режима работы
    print("\n" + "="*50)
    print("Выберите режим работы:")
    print("1 - Обучить новую модель")
    print("2 - Загрузить сохраненную модель")
    print("="*50)
    choice = input("Введите 1 или 2: ").strip()
    
    if choice not in ['1', '2']:
        print("Неверный выбор. Завершение работы.")
        return
    
    train_new_model = (choice == '1')
    model_dir = './models'
    model_path = os.path.join(model_dir, Config.MODEL_PATH)
    
    # Создаем папку models если ее нет
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Проверка доступности CUDA
    if Config.DEVICE == torch.device('cuda'):
        print(f"\nUsing CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Using {Config.NUM_WORKERS} workers for DataLoader.")
    else:
        print("\nWARNING: CUDA not available, using CPU. Training will be VERY slow.")
        Config.NUM_WORKERS = 0
        print("Setting num_workers to 0 for CPU usage.")

    # Загрузка данных
    print("\nLoading data...")
    try:
        df = pd.read_csv(Config.TRAIN_DATA)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {Config.TRAIN_DATA}")
        return

    # Уменьшение данных если нужно
    if Config.DATA_FRACTION < 1.0:
        print(f"Original training data size: {len(df)}")
        df = df.sample(frac=Config.DATA_FRACTION, random_state=Config.SEED)
        print(f"Reduced training data size: {len(df)} ({Config.DATA_FRACTION*100:.1f}%)")
    else:
        print(f"Using full training data ({len(df)} samples).")

    # Разделение на train/val
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=Config.SEED)
    del df

    # Построение словаря
    text_processor = TextProcessor()
    text_processor.build_vocab(
        pd.concat([train_df['abstract'], train_df['title']]),
        min_freq=Config.MIN_FREQ
    )
    
    # Проверка соответствия спецтокенов
    assert Config.PAD_IDX == text_processor.word2idx['<pad>']
    assert Config.SOS_IDX == text_processor.word2idx['<sos>']
    assert Config.EOS_IDX == text_processor.word2idx['<eos>']
    assert Config.UNK_IDX == text_processor.word2idx['<unk>']

    # Создание DataLoader
    print("\nCreating data loaders...")
    train_dataset = ArticleDataset(train_df['abstract'], train_df['title'], text_processor)
    val_dataset = ArticleDataset(val_df['abstract'], val_df['title'], text_processor)
    
    collate = partial(collate_fn, pad_idx=Config.PAD_IDX)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    # Инициализация модели
    print("\nInitializing model...")
    input_dim = output_dim = text_processor.n_words

    attn = Attention(Config.HID_DIM, Config.HID_DIM)
    enc = Encoder(input_dim, Config.ENC_EMB_DIM, Config.HID_DIM, Config.ENC_DROPOUT)
    dec = Decoder(output_dim, Config.DEC_EMB_DIM, Config.HID_DIM, Config.HID_DIM, Config.DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, Config.PAD_IDX, Config.SOS_IDX, Config.EOS_IDX, Config.DEVICE).to(Config.DEVICE)

    # Инициализация весов
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    model.apply(init_weights)
    
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)

    # Загрузка модели если выбрано
    if not train_new_model:
        try:
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            print(f"\nModel successfully loaded from {model_path}")
            # Пропускаем обучение
            test_model(model, text_processor)
            return
        except Exception as e:
            print(f"\nError loading model: {e}")
            print("Proceeding with training a new model")
            train_new_model = True

    # Обучение модели
    if train_new_model:
        best_valid_loss = float('inf')
        print(f"\nStarting training for {Config.N_EPOCHS} epochs...")

        for epoch in range(Config.N_EPOCHS):
            start_epoch_time = time.time()

            train_loss = train(model, train_loader, optimizer, criterion, Config.CLIP)
            valid_loss = evaluate(model, val_loader, criterion)

            end_epoch_time = time.time()
            epoch_mins, epoch_secs = divmod(end_epoch_time - start_epoch_time, 60)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_path)
                print(f"\nValidation loss improved. Model saved to {model_path}")

            print(f'\nEpoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # Тестирование модели
    test_model(model, text_processor)

    # Вывод общего времени выполнения
    end_total_time = time.time()
    total_mins, total_secs = divmod(end_total_time - start_total_time, 60)
    print(f"\nTotal execution time: {int(total_mins)}m {int(total_secs)}s")

# Запуск основной функции
if __name__ == '__main__':
    main()