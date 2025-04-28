import pytest
from fastapi.testclient import TestClient
from app import app
from PIL import Image
import io

client = TestClient(app)


@pytest.fixture
def generate_test_image():
    """Создаёт в памяти простой RGB-изображение"""
    def _generate():
        image = Image.new("RGB", (128, 128), color=(255, 0, 0))
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr

    return _generate


def test_predict_success(generate_test_image):
    """Проверяем, что успешный запрос возвращает 200 и есть 'results'."""
    img = generate_test_image()
    files = {"file": ("test.png", img, "image/png")}

    response = client.post("/predict/", files=files)

    assert response.status_code == 200
    assert "results" in response.json()
    assert isinstance(response.json()["results"], list)


def test_predict_no_file():
    """Проверяем, что без файла возвращается ошибка."""
    response = client.post("/predict/")

    assert response.status_code == 422


def test_predict_invalid_file():
    """Проверяем, что отправка не-изображения вызывает ошибку обработки."""
    files = {"file": ("test.txt", io.BytesIO(b"this is not an image"), "text/plain")}

    response = client.post("/predict/", files=files)

    assert response.status_code == 422 or response.status_code == 500
