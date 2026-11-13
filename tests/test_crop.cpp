#include "preprocessor.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: ./test_crop <caminho_da_imagem.jpg>\n";
        return 1;
    }

    // Carrega a imagem original (assumindo que o arquivo existe)
    cv::Mat frame = cv::imread(argv[1]);
    if (frame.empty()) {
        std::cerr << "Erro ao carregar a imagem.\n";
        return 1;
    }

    // Caixa de mentira simulando o rosto do motorista (ajuste para a sua imagem de teste)
    cv::Rect mock_bbox(100, 100, 150, 150); 

    // Executa a lógica do Gabriel
    cv::Mat tensor_input = get_driver_crop(frame, mock_bbox);

    // Converte de volta pra BGR só pra poder salvar e abrir no visualizador de imagens do Arch
    cv::Mat output_bgr;
    cv::cvtColor(tensor_input, output_bgr, cv::COLOR_RGB2BGR);
    
    cv::imwrite("resultado_crop.jpg", output_bgr);
    std::cout << "Processado! Shape final: " << tensor_input.cols << "x" << tensor_input.rows << " Canais: " << tensor_input.channels() << "\n";
    std::cout << "Imagem salva como resultado_crop.jpg\n";

    return 0;
}