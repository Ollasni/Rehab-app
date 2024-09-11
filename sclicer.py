import os
import cv2

def create_directory(path):
    """
    Создает директорию, если она не существует.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def slice_video(video_path, output_folder, segment_duration=5):
    """
    Нарезает видео на фрагменты длиной segment_duration секунд и сохраняет их.
    video_path: путь к видеофайлу
    output_folder: папка, куда будут сохранены нарезанные видеофрагменты
    segment_duration: продолжительность каждого фрагмента (в секундах)
    """
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    
    # Получаем частоту кадров и общее количество кадров
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Количество кадров в секунду
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Общее количество кадров
    segment_frames = fps * segment_duration  # Количество кадров в одном сегменте
    
    video_name = os.path.basename(video_path).split('.')[0]  # Имя видео без расширения
    
    # Создаем директорию для нарезанных фрагментов
    create_directory(output_folder)
    
    # Индекс текущего фрагмента
    segment_index = 0
    
    # Чтение кадров из видео
    current_frame = 0
    success, frame = cap.read()
    while success:
        # Путь для сохранения текущего фрагмента
        segment_output_path = os.path.join(output_folder, f"{video_name}_part{segment_index}.mp4")
        
        # Создаем видеофайл для записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек
        out = cv2.VideoWriter(segment_output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        
        for i in range(segment_frames):
            if not success or current_frame >= total_frames:
                break
            out.write(frame)  # Записываем кадр в видеофайл
            success, frame = cap.read()
            current_frame += 1
        
        out.release()  # Закрываем файл фрагмента
        segment_index += 1
    
    cap.release()
    print(f"Видео {video_name} успешно нарезано и сохранено в {output_folder}.")

def process_videos_in_folder(input_folder, output_base_folder, segment_duration=5):
    """
    Обрабатывает все видео в папке, нарезает их и сохраняет в отдельные папки.
    input_folder: папка с исходными видео
    output_base_folder: базовая папка для сохранения нарезанных фрагментов
    segment_duration: продолжительность каждого фрагмента (в секундах)
    """
    # Проходим по всем файлам в папке
    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)
        
        if os.path.isfile(video_path) and video_file.endswith(('.mp4', '.avi', '.mov')):
            # Папка для фрагментов текущего видео
            video_name = os.path.splitext(video_file)[0]
            output_folder = os.path.join(output_base_folder, video_name)
            
            # Нарезаем видео
            slice_video(video_path, output_folder, segment_duration)

# Путь к папке с видео
input_folder = '/home/olga/Pictures/VIDEO'

# Путь к папке, куда будут сохранены фрагменты
output_base_folder = '/home/olga/Pictures/VIDEO_scliced'

# Длительность каждого фрагмента (в секундах)
segment_duration = 5

# Запуск процесса нарезки всех видео
process_videos_in_folder(input_folder, output_base_folder, segment_duration)
