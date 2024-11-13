import java.util.ArrayList;
import java.util.List;

class Dataset {
    private List<double[]> features; //X
    private List<Integer> labels; //Y

    public Dataset() { // const vacío
        features = new ArrayList<>();
        labels = new ArrayList<>();
    }

    //añadir cada X y Y
    public void addData(double[] feature, int label) {
        features.add(feature);
        labels.add(label);
    }

    public List<double[]> getFeatures() {
        return features;
    }

    public List<Integer> getLabels() {
        return labels;
    }
}