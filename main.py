# Раздел импорта библиотек

import base64
import streamlit as st
import torch
from torch import nn
from torchvision import transforms, models
from dataclasses import dataclass
from PIL import Image

# Установка фонового изображения приложения

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('D:/Aquariumist/Изображения/1.png')

# Инициализация начальных состояний

if 'Model_load_complete' not in st.session_state:
    st.session_state['Model_load_complete'] = False
if 'selected_model_type'not in st.session_state:
    st.session_state['selected_model_type'] = None
if 'loading_model' not in st.session_state:
    st.session_state['loading_model'] = None
if 'using_model' not in st.session_state:
    st.session_state['using_model'] = None
if 'predicted_class' not in st.session_state:
    st.session_state['predicted_class'] = None
if 'loading_img' not in st.session_state:
    st.session_state['loading_img'] = None
if 'loaded_model_type' not in st.session_state:
    st.session_state['loaded_model_type'] = None

# Класс для реализации структуры слайдбокса

@dataclass
class Row:
    id: int
    name: str

    def __str__(self):
        return self.name

# Определение слайдбара для выбора типа загружаемой модели

selected = st.sidebar.selectbox("Выбирите тип модели и нажмите кнопку загрузки", (
    Row(0, "---- Тип модели не выбран ----"),
    Row(1, "Для мобильных устройств"),
    Row(2, "Для персонального компьютера"),
), help='Кликните по стрелке ниже и выберите тип классификатора, соответствующий Вашему устройству')

# Выбор и загрузка обученной модели классификации

if st.sidebar.button("Загрузить выбранную модель") and selected.id != 0:
    st.session_state.loading_model = True
    if st.session_state.loading_model == True:
        with st.sidebar.status("Загрузка выбранной модели...", expanded=True) as status:
            if selected.id == 1:
                model = models.mobilenet_v3_large()
                model.features[16][0] = nn.Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
                num_ftrs_0 = model.classifier[0].in_features
                model.classifier = nn.Sequential(nn.Linear(num_ftrs_0, 10))
                model.load_state_dict(
                    torch.load('D:/Aquariumist/Models/Для мобильных устройств/mobilenet_v3_large_Conv_and_last_Linear_new_My_Weights.pth',
                               map_location=torch.device('cpu'), weights_only=True))
                model.eval()
                st.session_state.loaded_model_type = 1
                st.session_state.using_model = model
            elif selected.id == 2:
                model = models.efficientnet_v2_l()
                model.features[8][0] = nn.Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
                num_ftrs_3 = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_ftrs_3, 10)
                model.load_state_dict(
                    torch.load('D:/Aquariumist/Models/Для ПК/efficientnet_v2_large_last_Conv_and_last_Linear_My_Weights.pth',
                               map_location=torch.device('cpu'), weights_only=True))
                model.eval()
                st.session_state.loaded_model_type = 2
                st.session_state.using_model = model
            status.update(label="Загрузка модели завершена!", state="complete", expanded=False)
            st.session_state.Model_load_complete = True
            st.session_state.selected_model_type = f'Модель {selected.name.lower()}'
            st.session_state.loading_model = False
if st.session_state.Model_load_complete == False:
    st.sidebar.markdown(
        ":orange-badge[⚠️ Выберите и загрузите модель] "
    )

# Разбиение области страницы на 2 колонки

col1, col2 = st.columns([1, 2], gap="large")

# Основной код

if st.session_state.Model_load_complete == True:
    st.sidebar.markdown(
        f":violet-badge[:material/preliminary: {st.session_state.selected_model_type}]"
    )
    uploaded_file = st.sidebar.file_uploader("Загрузите фото растения", type=["jpg", "jpeg", "png"], help='Для определения вида растения нажмите кнопку "Browse files" и загрузите изображение растения в одном из трех форматов: jpg, jpeg, png')
    if uploaded_file:
        image = Image.open(uploaded_file)
        image.save("D:/Aquariumist/Изображения/Тестовое изображение/output_image.jpg")
        st.sidebar.image("D:/Aquariumist/Изображения/Тестовое изображение/output_image.jpg", width=224)
        test_img = Image.open('D:/Aquariumist/Изображения/Тестовое изображение/output_image.jpg')
        if st.session_state.loaded_model_type == 1:
            test_img = transforms.Resize((232, 232))(test_img)
            test_img = transforms.ToTensor()(test_img)
            test_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(test_img)
            test_img = torch.reshape(test_img, (1, 3, 232, 232))
            st.session_state.loading_img = True
        elif st.session_state.loaded_model_type == 2:
            test_img = transforms.Resize((480, 480))(test_img)
            test_img = transforms.ToTensor()(test_img)
            test_img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(test_img)
            test_img = torch.reshape(test_img, (1, 3, 480, 480))
            st.session_state.loading_img = True
        classes = ['Anubias barteri var Glabra', 'Anubias barteri var nana Petite',
                   'Anubias barteri var nana Round Leaf (Coin Leaf)', 'Cabomba caroliniana var caroliniana',
                   'Elodea canadensis', 'Hemianthus callitrichoides Cuba', 'Hygrophila difformis',
                   'Nymphoides hydrophylla Taiwan', 'Vallisneria spiralis', 'Vesicularia dubyana']
        class_names = {}
        for key, value in enumerate(classes):
            class_names[key] = value
        class_links = {'Anubias barteri var Glabra':{'Блог Tetra.net': 'https://blog.tetra.net/ru/ru/anubias-bartera-zhestkolistnyj-lyubimchik',
                                                     'Описание на сайте Aqa.ru': 'https://www.aqa.ru/wiki/Анубиас_глабра',
                                                     'Описание вида на Википедии': 'https://ru.wikipedia.org/wiki/Анубиас_Бартера',
                                                     'Форум Flowgrow.de': 'https://www.flowgrow.de/db/aquaticplants/anubias-barteri-var-glabra'},
                       'Anubias barteri var nana Petite': {'Блог Tetra.net': 'https://blog.tetra.net/ru/ru/anubias-bartera-zhestkolistnyj-lyubimchik',
                                                           'Описание на сайте Aqa.ru': 'https://www.aqa.ru/wiki/Анубиас_нана-петит',
                                                           'Описание вида на Википедии': 'https://ru.wikipedia.org/wiki/Анубиас_Бартера',
                                                           'Форум Flowgrow.de': 'https://www.flowgrow.de/db/aquaticplants/anubias-barteri-var-nana-petite-bonsai'},
                       'Anubias barteri var nana Round Leaf (Coin Leaf)': {'Блог Tetra.net': 'https://blog.tetra.net/ru/ru/anubias-bartera-zhestkolistnyj-lyubimchik',
                                                           'Описание на сайте Aqa.ru': 'https://www.aqa.ru/wiki/Анубиас_глабра/Анубиас_монетка',
                                                           'Описание вида на Википедии': 'https://ru.wikipedia.org/wiki/Анубиас_Бартера',
                                                           'Форум Flowgrow.de': 'https://www.flowgrow.de/db/aquaticplants/anubias-barteri-var-nana-round-leaf'},
                       'Cabomba caroliniana var caroliniana': {'Блог Tetra.net': 'https://blog.tetra.net/ru/ru/kabomba-dlinnostebelnaya-krasavica',
                                                               'Описание на сайте Aqa.ru': 'https://www.aqa.ru/wiki/Кабомба_каролинская',
                                                               'Описание вида на Википедии': 'https://ru.wikipedia.org/wiki/Кабомба_каролинская',
                                                               'Форум Flowgrow.de': 'https://www.flowgrow.de/db/aquaticplants/cabomba-caroliniana-var-caroliniana'},
                       'Elodea canadensis': {'Блог Tetra.net': 'https://blog.tetra.net/ru/ru/ehlodeya-kanadskaya',
                                             'Описание на сайте Aquamir63.ru': 'https://aquamir63.ru/publ/akvariumnye_rastenija/rasteniya_na_eh/ehlodeja_kanadskaja_elodea_canadensis/50-1-0-171',
                                             'Описание вида на Википедии': 'https://ru.wikipedia.org/wiki/Элодея_канадская',
                                             'Форум Flowgrow.de': 'https://www.flowgrow.de/db/aquaticplants/elodea-canadensis'},
                       'Hemianthus callitrichoides Cuba': {'Блог Tetra.net': 'https://blog.tetra.net/ru/ru/hemiantus-kuba-lyubimec-akvaskejperov',
                                                           'Описание на сайте Aqa.ru': 'https://www.aqa.ru/Hemianthus_callitrichoides',
                                                           'Описание вида на Википедии': 'https://ru.wikipedia.org/wiki/Micranthemum_callitrichoides',
                                                           'Форум Flowgrow.de': 'https://www.flowgrow.de/db/aquaticplants/hemianthus-callitrichoides-cuba'},
                       'Hygrophila difformis': {'Блог Tetra.net': 'https://blog.tetra.net/ru/ru/gigrofila-sinema-menyayushchaya-listya',
                                                'Описание на сайте Aqa.ru': 'https://www.aqa.ru/wiki/Гигрофила_разнородная',
                                                'Описание вида на Википедии': 'https://ru.wikipedia.org/wiki/Гигрофила_разнолистная',
                                                'Форум Flowgrow.de': 'https://www.flowgrow.de/db/aquaticplants/hygrophila-difformis'},
                       'Nymphoides hydrophylla Taiwan': {'Блог Tetra.net': 'https://blog.tetra.net/ru/ru/nimfoides-flipper',
                                                           'Описание на сайте Housaqua.com': 'https://housaqua.com/2689-nimfoides-flipper.html',
                                                           'Описание вида на Википедии': 'https://en.wikipedia.org/wiki/Nymphoides_hydrophylla',
                                                           'Форум Flowgrow.de': 'https://www.flowgrow.de/db/aquaticplants/nymphoides-sp-taiwan-flipper'},
                       'Vallisneria spiralis': {'Блог Tetra.net': 'https://blog.tetra.net/ru/ru/vallisneriya-spiralnaya-listya-s-zavitkami',
                                                'Описание на сайте Aqa.ru': 'https://www.aqa.ru/vallisneria_spiralis',
                                                'Описание вида на Википедии': 'https://ru.wikipedia.org/wiki/Валлиснерия_спиральная',
                                                'Форум Flowgrow.de': 'https://www.flowgrow.de/db/aquaticplants/vallisneria-spiralis'},
                       'Vesicularia dubyana': {'Блог Tetra.net': 'https://blog.tetra.net/ru/ru/yavanskij-moh-priyut-dlya-malkov',
                                               'Описание на сайте Aqa.ru': 'https://www.aqa.ru/ptichka_mox',
                                               'Описание вида на Википедии': 'https://ru.wikipedia.org/wiki/Яванский_мох',
                                               'Форум Flowgrow.de': 'https://www.flowgrow.de/db/aquaticplants/vesicularia-dubyana'},
                       }
        with st.sidebar.status("Определение вида растения...", expanded=True) as status:
            st.session_state.predicted_class = class_names[
                torch.max(st.session_state.using_model(test_img), 1).indices.tolist()[0]]
            status.update(label="Вид растения определен!", state="complete", expanded=False)
            st.session_state.loading_img = False
        with st.empty():
            with col1:
                st.markdown('', unsafe_allow_html=True)
                st.header("")
                st.header(f"")
                st.image(f"D:/Aquariumist/Изображения/Тестовое изображение/Img/standard/{st.session_state.predicted_class}.jpg")
            with col2:
                st.markdown('', unsafe_allow_html=True)
                st.header(f"")
                st.header(f"{st.session_state.predicted_class}")
                for key, value in class_links[st.session_state.predicted_class].items():
                    st.html(f"<p><font size=4>  - {key} <a href={value} target='_blank'> ссылка </a>. <font></p>")