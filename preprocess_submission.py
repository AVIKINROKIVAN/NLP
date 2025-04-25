import pandas as pd

def preprocess_submission(input_file, output_file):
    # Читаем файл
    df = pd.read_csv(input_file)
    
    # Заменяем NaN на пустые строки
    df['title'] = df['title'].fillna('')
    
    # Сохраняем обработанный файл
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    preprocess_submission('submission_generated.csv', 'submission_processed.csv')