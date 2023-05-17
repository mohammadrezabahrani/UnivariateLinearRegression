#include <iostream>
#include <iterator> // Include the <iterator> library for begin() and end()
#include <cmath>
using namespace std;

class LinearRegression {

    public:

        void Set_X_Data(double* x_data, int x_size,double* y_data,int y_size,bool normaldata = false) {

            x = x_data;
            x_len = x_size;

            y = y_data;
            y_len = y_size;
           
            if(normaldata){ Normalize(); }

            calculate_hat();
            calculate_sum_x_hat_y_hat();
            calculate_beta1();
            calculate_beta0();
            predict_for_calculate_errors();
            calculate_MAE();
            calculate_MSE();
            calculate_RMSE();
        }

        double* getX() { return x; }
        double* getXHat() { return x_hat; }
        int getLenX() { return x_len; }

        double* getY() { return y; }
        double* getYHat() { return y_hat; }
        int getLenY() { return y_len; }
        
        double getBeta1() { return beta1; }
        double getBeta0() { return beta0; }


        double *getYPredict() { return y_predict; }


        double getMAE() { return MAE; }
        double getMSE() { return MSE; }
        double getRMSE() { return RMSE; }


        double predict(double x) { return beta0 + beta1 * x; }

    private:
        
        double* x = nullptr;
        double* y = nullptr;
        int x_len = 0;
        int y_len = 0;

        double *x_hat = nullptr;
        double *y_hat = nullptr;
        double *x_hat_squared = nullptr;

        double x_hat_squared_sum = 0;
        double y_hat_x_hat_sum = 0;
        
        double beta0 = 0;
        double beta1 = 0;
        
        double *y_predict = nullptr;

        double MAE = 0;
        double MSE = 0;
        double RMSE = 0;


        double findMin(double arr[], int size) {
            double minVal = arr[0];
            for (int i = 1; i < size; i++) {
                if (arr[i] < minVal) { minVal = arr[i];}
            }
            return minVal;
        }

        double findMax(double arr[], int size) {
            double maxVal = arr[0];
            for (int i = 1; i < size; i++) {
                if (arr[i] > maxVal) { maxVal = arr[i];}
            }
            return maxVal;
        }

        void Normalize()
        {

            double x_min = findMin(x,x_len);
            double x_max = findMax(x,x_len);
            double y_min = findMin(y,y_len);
            double y_max = findMax(y,y_len);

            for(int index=0;index<x_len;index++)
            {
                x[index] = (x[index] - x_min) / (x_max - x_min);
            }
            for(int index=0;index<y_len;index++)
            {
                y[index] = (y[index] - y_min) / (y_max - y_min);
            }
        }
        
        double avg_x()
        {          
            double sum=0;  
            for(int i=0;i<x_len;i++) { sum+=x[i];}
            return sum / x_len;
        }

        double avg_y()
        {          
            double sum=0;  
            for(int i=0;i<y_len;i++) { sum+=y[i];}
            return sum / y_len;
        }

        void calculate_hat()
        {
            x_hat = new double[x_len];
            y_hat = new double[y_len];
        
            double avgx = avg_x();
            double avgy = avg_y();
            for(int i=0;i<x_len;i++) { x_hat[i] = x[i] - avgx;}
            for(int i=0;i<y_len;i++) { y_hat[i] = y[i] - avgy;}
            for(int i=0;i<y_len;i++) { y_hat_x_hat_sum += x_hat[i] * y_hat[i];}

        }

        void calculate_sum_x_hat_y_hat()
        {
            x_hat_squared = new double[x_len];
            x_hat_squared_sum = 0;
            for(int i=0; i<x_len; i++) 
            { 
                x_hat_squared[i] = x_hat[i] * x_hat[i];
                x_hat_squared_sum += x_hat_squared[i];
            }
        }


        void calculate_beta1()
        {
            beta1 =  y_hat_x_hat_sum /x_hat_squared_sum ;
        }

        void calculate_beta0()
        {
            beta0 =  avg_y() - beta1 * avg_x();
        }

       

        void predict_for_calculate_errors()
        {
            y_predict = new double[y_len];
            for(int i=0;i<x_len;i++) { y_predict[i] = beta0 + beta1 * x[i];}
        }

        void calculate_MAE()
        {
            double *y_mae = new double[y_len];
            double sum_mae = 0;
            for(int i=0;i<y_len;i++)
            {
                y_mae[i] = std::abs(y[i] - y_predict[i]);
                sum_mae += y_mae[i];
            }
            MAE = sum_mae / y_len;
        }

        void calculate_MSE()
        {
            double *y_mse = new double[y_len];
            double sum_mse = 0;
            for(int i=0;i<y_len;i++)
            {
                y_mse[i] = (y[i] - y_predict[i]) * (y[i] - y_predict[i]);
                sum_mse += y_mse[i];
            }
            MSE = sum_mse / y_len;
        }

        void calculate_RMSE()
        {
           RMSE =  std::sqrt(MSE);
        }
};

int main() {
    cout<<"****************** START *****************" <<endl<<endl;


    double x[] = { 230, 195, 220, 180, 200, 185, 240, 199 };
    double y[] = { 170, 110, 160, 150, 125, 115, 180, 130 };

    int sizex = sizeof(x) / sizeof(x[0]);
    int sizey = sizeof(y) / sizeof(y[0]);

    LinearRegression res;

    res.Set_X_Data(x, sizex,y,sizey,false); 
    cout<< res.predict(230)<<endl;

    cout<<res.getMAE();

    cout<<endl<<"******************* END ******************" <<endl;
    return 0;
}
