#Edge detection

Edge detection operators can be compared in an objective way. The performance of an edge detection operator in noise can be measured quantitatively as follows: Let n0 be the number of edge pixels declared and n1 be number of missed or new edge pixels after adding noise. If n0 is held fixed for the noiseless as well as noisy images, then the edge detection error rate is Pe=n1/n0
Compare the performance of the gradient operators of Roberts, Sobel, Prewitt and the 5x5 stochastic gradient on a noisy image with SNR= 8dB. 
Note that the pixel location (m,n) is declared an edge location if the magnitude gradient g(m,n) exceeds a THRESH value of 150. The edge locations constitute an edge map. For this assignment, you can select 512x512 BMP or RAW grayscale image of Lena.
