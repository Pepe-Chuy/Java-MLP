import java.util.Random;

public class MLPHardcode {

    // Arquitectura (3 capas, dos matriciales(entrada-oculta) y una lista(output)
    //primer capa input->oculta
    double[][] inputToHiddenWeights = new double[784][128]; // 784 (tamaño de las imagenes) 128 neuronas (tamaño de la siguiente capa)

    //segunda capa oculta->output
    double[][] hiddenToOutputWeights = new double[128][10]; // 128 neuronas, 10 outputs para la siguiente capa
    double[] hiddenLayerBias = new double[128]; // un sezgo para cada neurona
    double[] hiddenLayerOutputs = new double[128];

    //tercera capa output
    double[] outputLayerBias = new double[10]; //un sezgo por neurona(clase)
    double[] outputLayer = new double[10];


    // inicializa los pesos random entre -0.5 y 0.5 para converger más rapido
    public static void initWeights(double[][] matrix) {
        Random random = new Random(); //instanciamos para generar los pesos random
        for (int i = 0; i < matrix.length; i++) { // itera las filas de la matriz
            for (int j = 0; j < matrix[i].length; j++) { //itera las columnas
                matrix[i][j] = -0.5 + random.nextDouble(); //asigna el valor random
            }
        }
    }

    // forward prop
    public double[] forward(double[] inputVector) { /// HAY QUE GENERALIZAR PARA N CAPAS
        // primer capa input->oculta
        for (int i = 0; i < hiddenLayerOutputs.length; i++) { //itera las neuronas de la capa oculta
            hiddenLayerOutputs[i] = hiddenLayerBias[i]; //inicializa el output y sezgo
            for (int j = 0; j < inputVector.length; j++) { // iterar el input
                hiddenLayerOutputs[i] += inputVector[j] * inputToHiddenWeights[j][i]; // sumar el input y el peso
            }
            hiddenLayerOutputs[i] = activationFunction(hiddenLayerOutputs[i]); //aplicar activación
        }

        // segunda capa oculta->output
        for (int i = 0; i < outputLayer.length; i++) { // iterar las neuronas de la capa de salida
            outputLayer[i] = outputLayerBias[i]; // inicializa el output y sezgo
            for (int j = 0; j < hiddenLayerOutputs.length; j++) { // iterar el output
                outputLayer[i] += hiddenLayerOutputs[j] * hiddenToOutputWeights[j][i]; // sumar el input y el peso
            }
            outputLayer[i] = activationFunction(outputLayer[i]); //aplicar activación
        }
        return outputLayer; // outputs de la capa de salida
    }

    ///TODAS LAS FUNCIONES TIENEN QUE SER ESTÁTICAS
    // Funcion de activación Sigmoide
    public static double activationFunction(double x) { //recibir x
        return 1 / (1 + Math.exp(-x)); // aplicar Sigmoide
    }

    // Funcion de costo MSE
    public static double costFunction(double[] predicted, double[] target) { // recibe predicción y target como arrays
        int n = predicted.length;
        double error = 0;
        for (int i = 0; i < n; i++) { // iterar las predicciones
            error += Math.pow(predicted[i] - target[i], 2); //sumar el error
        }
        return error / n; //promediarlo
    }

    // Función de precisión
    public static boolean isPredictionCorrect(double[] predicted, int targetIndex) {
        int predictedIndex = 0;
        for (int i = 1; i < predicted.length; i++) { //iterar las predicciones
            if (predicted[i] > predicted[predictedIndex]) { // si la prediccion actual es mejor que la pasada
                predictedIndex = i; //cambiar la mejor prediccion
            }
        }
        return predictedIndex == targetIndex; // solo regresar para el indice del target
    }

    // Backprop
    //recibe prediccion, target, tasa de aprendizaje y el input
    public void backprop(double[] predicted, double[] target, double learningRate, double[] inputVector) {
        ///(vamos a necesitar getters en estos dos cuando abstraigamos)
        int outputSize = predicted.length; //tamaño del output
        int hiddenSize = hiddenToOutputWeights.length; // tamaño de la capa oculta

        // Calcular el error de la salida
        /// HAY QUE GENERALIZARLA TAMBIÉN
        double[] outputErrors = new double[outputSize]; //guardar los errores
        for (int i = 0; i < outputSize; i++) { //iteramos las salidas
            outputErrors[i] = predicted[i] - target[i]; // calcular el error
        }

        // Calcular el error de la oculta
        double[] hiddenErrors = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) { //iterar las neuronas
            hiddenErrors[i] = 0;
            for (int j = 0; j < outputSize; j++) { //iterar el output
                hiddenErrors[i] += outputErrors[j] * hiddenToOutputWeights[i][j]; //error x peso
            }
            /// HAY QUE GENERALIZAR EN ACTIVACION
            hiddenErrors[i] *= hiddenLayerOutputs[i] * (1 - hiddenLayerOutputs[i]); // derivada de la activación (sigmoide)
        }

        /// GENERALIZAR COMO METODO DE LAS CAPAS (POSIBLE CLASE PESOS???)
        // Actualizar pesos
        for (int i = 0; i < hiddenSize; i++) { //iterar las neuronas de la oculta
            for (int j = 0; j < outputSize; j++) { //iterar el output
                hiddenToOutputWeights[i][j] -= learningRate * outputErrors[j] * hiddenLayerOutputs[i]; // el peso es igual a la taza por el error por el output (o la capa de la izquierda)
            }
        }

        /// TAMBIÉN FALTA GENERALZAR ACTUALIZAR SEZGOS EN LAS NEURONAS
        // Actualizar sezgo de la capa de salida
        for (int i = 0; i < outputSize; i++) {
            outputLayerBias[i] -= learningRate * outputErrors[i]; // el sezgo le restamos el error por la tasa
        }

        // Actualizar pesos de la oculta
        for (int i = 0; i < inputVector.length; i++) {
            for (int j = 0; j < hiddenSize; j++) { //iterar las neuronas de la capa oculta
                inputToHiddenWeights[i][j] -= learningRate * hiddenErrors[j] * inputVector[i]; // el peso es igual a la taza por el error por el output (o la capa de la izquierda)
            }
        }

        // Actualizar sezgo de la capa oculta
        for (int i = 0; i < hiddenSize; i++) {
            hiddenLayerBias[i] -= learningRate * hiddenErrors[i]; // el sezgo le restamos el error por la tasa
        }
    }

    // EL MAIN GOOOD
    public static void main(String[] args) {
        MLPHardcode nn = new MLPHardcode();

        // Cargar el datasset
        Dataset trainData = MNISTLoader.loadTrain();
        double learningRate = 0.01; // LR
        int epochs = 2; // epocas

        // inicializar los pesos (n (capas) -1 )
        initWeights(nn.inputToHiddenWeights);
        initWeights(nn.hiddenToOutputWeights);

        for (int epoch = 0; epoch < epochs; epoch++) { //iterar las epocas
            double totalCost = 0; // costo total
            int correctPredictions = 0; // predicciones correctas

            for (int i = 0; i < trainData.getFeatures().size(); i++) { //iterar el train
                double[] inputVector = trainData.getFeatures().get(i); // obetener el vector de entrada
                double[] target = new double[10]; // One-hot encoding para las  10 clases
                target[trainData.getLabels().get(i)] = 1; //indice 1

                // Forward pass
                double[] output = nn.forward(inputVector); //forward prop
                double cost = costFunction(output, target); //obtener el costo
                totalCost += cost; //sumar el costo


                /// HAY QUE GENERALIZAR
                if (isPredictionCorrect(output, trainData.getLabels().get(i))) { // checar si la prediccion es verdadera
                    correctPredictions++; //sumar predicciones correectas
                }

                // Bakcprop
                nn.backprop(output, target, learningRate, inputVector);
            }

            // Imprimir el costo promedio y accuracy para cada época
            double averageCost = totalCost / trainData.getFeatures().size(); //costo promedio
            double accuracy = (double) correctPredictions / trainData.getFeatures().size() * 100; //Accuracy

            System.out.println("Epoch: " + (epoch + 1) + "/" + epochs + // imprimir epoca
                    " | Average Cost: " + averageCost + // imprimir costo promedio
                    " | Accuracy: " + accuracy + "%"); // imprimir accuracy
        }
    }
}
