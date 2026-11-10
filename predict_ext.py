import models_ext
import predict

predict.build_model = models_ext.build_model

if __name__ == "__main__":
    predict.main()
