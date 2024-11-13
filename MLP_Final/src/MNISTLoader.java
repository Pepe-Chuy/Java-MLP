import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class MNISTLoader {

    //Cargar el train tienen que ser estáticos para compilar desde antes
    public static Dataset loadTrain() {
        return read_csv("data/mnist_train.csv");
    }
    //Cargar el test
    public static Dataset loadTest() {
        return read_csv("data/mnist_test.csv");
    }

    //Leer CSV
    public static Dataset read_csv(String filePath) { //recibe el path
        Dataset dataset = new Dataset(); //inicializa un dataset vacío
        String line; // para guardar temporalmente cada línea
        String delimiter = ","; //splitteamos por las comas

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            br.readLine(); // Tirar el header

            // leer cada linea
            while ((line = br.readLine()) != null) {
                String[] values = line.split(delimiter); // cada separación se añade en una lista
                int label = Integer.parseInt(values[0]); // extraer el primer valor que es el label y hacerlo int
                double[] features = new double[values.length - 1]; //hacer el array de las imagenes

                for (int i = 1; i < values.length; i++) {
                    features[i - 1] = Double.parseDouble(values[i]); //hacer las imagenes double y guardarlo en features
                }

                dataset.addData(features, label); //llenar el dataset vacío
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dataset; //devolver el dataset
    }
}