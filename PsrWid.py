import numpy as np
import matplotlib.pyplot as plt
import george
from george import kernels
#plt.style.use('seaborn-dark')

# Things to be checked 
# on pulse bins
# remember you have plotting option for num_samples
# make I self.I
# use inheritance for the class
# remember - sadly, on_pulse_bins are correspond to the centered profile (phase shifted)
# orientation of log-polar plot - 180 upwards
#plotting should be I_fit+yerr and I_fit-yerr & not I_fit+std and I_fit-std?

class PsrWid:
    def __init__(self, I):
        self.I = I


    def get_pulse_at_center(self) -> np.array:
            
            """
            Get the pulse profile at the center.
    
            Parameters:
            - I                 : np.array, the input array.
    
            Returns:
            - pulse_at_center   : np.array, the pulse profile at the center.
            """
    
            center_bin = int(len(self.I) / 2)
            peak_bin = np.argmax(self.I)
            pulse_at_center = np.roll(self.I, center_bin - peak_bin)

            return pulse_at_center


    def norm_onezero(self) -> np.array:

        """
        Specifically used for Pulsar's profile data.
        Normalize the input array to the range [0, 1].

        Parameters:
        - I: np.array, the input array.

        Returns:
        - norm_array: np.array, the normalized array.
        """
        centered_I = self.get_pulse_at_center()
        norm_array = (centered_I - np.min(centered_I)) / (np.max(centered_I) - np.min(centered_I))
        min1 = np.argmin(norm_array)
        norm_array = np.delete(norm_array, np.argmin(norm_array))
        min2 = np.argmin(norm_array)
        # adding deleted min1 = min2
        norm_array = np.insert(norm_array, min2, 0)

        return norm_array

    
    def get_off_pulse(self, 
                      on_pulse_bins=(400, 600), 
                      normalized=True) -> np.array:
       
        """
        Get the off-pulse region of the pulse profile.

        Parameters:
        - I             : np.array, the pulse profile.
        - on_pulse_bins : tuple, the bins of the on-pulse region.

        Returns:
        - off_pulse     : np.array, the off-pulse region.
        """

    
        if normalized:
            I = self.norm_onezero()
        else:
            I = self.I

        off_pulse = np.delete(I, np.arange(on_pulse_bins[0], on_pulse_bins[1]))

        return off_pulse
    

    def get_SNR(self, 
                on_pulse_bins=None, 
                normalized=True) -> float:
        
        """
        Calculate the signal-to-noise ratio (SNR) of the pulse profile.
        If the on_pulse_bins is not provided, 
        it by default takes the bins from 400 to 600.

        Parameters:
        - I             : np.array, the pulse profile.
        - on_pulse_bins : tuple, the bins of the on-pulse region.

        Returns:
        - SNR           : float, the signal-to-noise ratio.
        """

        # NORMALIZED SNR
        if normalized:
            I = self.norm_onezero()
            if on_pulse_bins is not None:                                      # on_pulse_bins are provided
                off_pulse = self.get_off_pulse(on_pulse_bins, normalized=True)
            else:                                                              # on_pulse_bins are not provided - default (400, 600
                off_pulse = self.get_off_pulse(normalized=True)

        # NON-NORMALIZED SNR
        else:
            I = self.I
            if on_pulse_bins is not None:
                off_pulse = self.get_off_pulse(on_pulse_bins, normalized=False)
            else:
                off_pulse = self.get_off_pulse(normalized=False)


        peak = np.max(I)
        std = np.std(off_pulse)
        mean = np.mean(off_pulse)
        SNR = (peak - mean) / std

        return SNR
    

    def Widths_at_Threshold(self, 
                            X, Y, 
                            threshold, 
                            all_info=False) -> np.array:
         
        """
        Find the widths of the regions where the Gaussian Process prediction is above the threshold.

        Parameters:
        - X         : np.array, the input data.
        - Y         : np.array, the predicted values from the Gaussian Process.
        - threshold : float, the threshold value above which the regions are to be found.
        - all_info  : bool, if True, return the start and end points of widths.
                      if False, return the widths only.

        Returns:
        - widths    : np.array, the widths of the regions where the Gaussian Process prediction is above the threshold.
        """

        # Find the indices where the fit is above the threshold
        indices_above = np.where(Y >= threshold)[0]
        if len(indices_above) == 0:
            return []

        # Find the start and end points of the widths - fit is above threshold
        widths = []
        start = indices_above[0]
        for i in range(1, len(indices_above)):
            if indices_above[i] != indices_above[i-1] + 1:
                end = indices_above[i-1]
                widths.append((X[start], X[end], X[end] - X[start]))
                start = indices_above[i]

        end = indices_above[-1]

        widths.append((X[start], X[end], X[end] - X[start]))

        if all_info:
            return np.array(widths) # return the start and end points of widths
        else:
            return np.array([width[2] for width in widths]) # return the widths only


    def GP_regression(self, 
                      length_scale=None, 
                      amplitude=None, 
                      on_pulse_bins=None, 
                      num_samples=None) -> np.array:
        
        """
        Perform Gaussian Process regression on the input data.

        Parameters:
        - I             : np.array, the input data.
        - length_scale  : float, the length scale of the RBF kernel.
        - amplitude     : float, the amplitude of the RBF kernel.
        - on_pulse      : tuple, the on-pulse region. 
                          If None, the empirical noise is calculated from the whole data.
        - num_samples   : int, the number of sample fits to plot.

        Returns:
        - mu            : np.array, the predicted values from the Gaussian Process (GP fit).
        - std           : np.array, the standard deviation of the GP fit.
        - yerr          : float, retrieved std of the white noise.
        - samples       : np.array, the sample fits from the GP.
        """

        # Hyperparameters
        if length_scale is None:
            length_scale = (2*np.pi/len(self.I)) + 0.1 * 2*np.pi/len(self.I)  # Typically around 0.0067 for 1024 bins
        else:
            length_scale = length_scale
        if amplitude is None:
            amplitude = 1.0   # Works well for normalized data
        else:
            amplitude = amplitude


        phi = np.linspace(0, 2*np.pi, len(self.I))
        y = self.norm_onezero()
        X = phi[:, np.newaxis]  # Reshape for compatibility

        # RBF kernel --> Squared Exponential Kernel
        kernel = amplitude**2 * kernels.ExpSquaredKernel(metric=length_scale)

        # yerr: empirical error in the data --> Retrieving the STD of the noise.
        if on_pulse_bins is not None:
            off_pulse = self.get_off_pulse(on_pulse_bins, normalized=True)
            emp_noise = np.std(np.diff(off_pulse))
        else:
            off_pulse = self.get_off_pulse(normalized=True)
            emp_noise = np.std(np.diff(y))

        yerr = emp_noise / np.sqrt(2)

        # Initialize the GP with the kernel
        gp = george.GP(kernel)
        gp.compute(X, yerr=yerr)  # Use the noise level as the y-error

        # Test points
        test_points = 2*len(self.I)  # ~2000 points works well for 1024 bins
        X_test = np.linspace(0, 2*np.pi, test_points)[:, np.newaxis]

        # Predictions
        mu, cov = gp.predict(y, X_test, return_cov=True)
        std = np.sqrt(np.diag(cov))

        if num_samples is not None:
            # Sample from the multivariate normal distribution to get possible fits
            samples = np.random.multivariate_normal(mu, cov, num_samples)
            return mu, std, yerr, samples
        else:
            return mu, std, yerr


    def get_knee_point(self, I_fit=None, bins=500, all_params=False) -> float:

        """
        Get the knee point of the input data using the Kneedle Algorithm.

        Parameters:
        - I_fit    : np.array, any array whose knee point is to be found.
        - bins     : int, the number of bins for the histogram.

        Returns:
        - knee_point: float, the knee point of the input data.
        """

        if I_fit is None:
            I_fit, std, yerr = self.GP_regression()
        else:
            I_fit = I_fit
        
        log_I = np.log(I_fit)
        phi = np.linspace(0, 2*np.pi, len(I_fit))

        # Histogram of the log(I)
        entries, edges, patches = plt.hist(log_I, bins=bins, density=True, cumulative=True, histtype='step')
        plt.close()

        xc = edges[:-1]
        yc = entries
        x_norm = (xc - xc.min()) / (xc.max() - xc.min())
        y_norm = (yc - yc.min()) / (yc.max() - yc.min())

        start_point = (x_norm[0], y_norm[0])
        end_point = (x_norm[-1], y_norm[-1])

        #distances = -((end_point[1] - start_point[1]) * x_norm - (end_point[0] - start_point[0]) * y_norm + end_point[0] * start_point[1] - end_point[1] * start_point[0] / np.sqrt((end_point[1] - start_point[1])**2 + (end_point[0] - start_point[0])**2))
        distances = y_norm - x_norm

        knee_point = np.argmax(distances)
        unnormalized_knee_point = xc[knee_point]

        if all_params:
            return unnormalized_knee_point, xc, yc, distances
        else:
            return unnormalized_knee_point


    def get_widths(self, 
                   wid_50=False, 
                   wid_10=False, 
                   wid_1=False, 
                   get_error=False,
                   all_info=False) -> np.array:
        """
        Get the widths of the pulse profile at 50%, 10%, and 1% of the peak intensity.

        Parameters:
        - I     : np.array, the input data.
        - wid_50: bool, if True, return the width at 50% of the peak intensity.
        - wid_10: bool, if True, return the width at 10% of the peak intensity.
        - wid_1 : bool, if True, return the width at 1% of the peak intensity.
        - all_info: bool, if True, (also) return the start and end points of the widths.

        Returns:
        - widths: np.array, the widths of the pulse profile at 50%, 10%, and 1% of the peak intensity.
        """

        I_fit, std, yerr = self.GP_regression()
        phi_fit = np.linspace(0, 2*np.pi, len(I_fit))
        peak = np.max(I_fit)

        widths = []

        # Widths derived from PsrWid Algorithm
        threshold = self.get_knee_point()
        threshold = np.exp(threshold)
        widths.append(self.Widths_at_Threshold(phi_fit, I_fit, threshold, all_info=all_info))

        # W50, W10, W1
        if wid_50:
            widths.append(self.Widths_at_Threshold(phi_fit, I_fit, peak * 0.5, all_info=all_info))
        if wid_10:
            widths.append(self.Widths_at_Threshold(phi_fit, I_fit, peak * 0.1, all_info=all_info))
        if wid_1:
            widths.append(self.Widths_at_Threshold(phi_fit, I_fit, peak * 0.01, all_info=all_info))

        # Error Calculation

        if get_error:
            threshold_plus = self.get_knee_point(I_fit + std)
            threshold_plus = np.exp(threshold_plus)
            widths_plus = self.Widths_at_Threshold(phi_fit, I_fit, threshold_plus, all_info=True)

            threshold_minus = self.get_knee_point(I_fit - std)
            threshold_minus = np.exp(threshold_minus) 
            widths_minus = self.Widths_at_Threshold(phi_fit, I_fit, threshold_minus, all_info=True)

            return np.array(widths), np.array(widths_plus), np.array(widths_minus), threshold_plus, threshold_minus
        
        else:
            return np.array(widths)

    
    def plot_results(self, 
                     on_pulse_bins=None, 
                     length_scale=None, 
                     amplitude=None, 
                     num_samples=None, 
                     bins=500, 
                     PSR_name=None, 
                     get_error=False,
                     xlim=(0, 2*np.pi)):
        
        """
        Plot the pulse profile, the log-polar profile, and the knee detection results.

        Parameters:
        - I             : np.array, the input data.
        - on_pulse_bins : tuple, the bins of the on-pulse region.
        - length_scale  : float, the length scale of the RBF kernel.
        - amplitude     : float, the amplitude of the RBF kernel.
        - num_samples   : int, the number of sample fits to plot.
        - bins          : int, the number of bins for the histogram.
        - PSR_name      : str, the name of the pulsar.
        - xlim          : tuple, the x-axis limits.

        Returns:
        PLOT 1: Pulse Profile and GP Fit
        PLOT 2: Log-Polar Profile
        PLot 3: Kneedle Algorithm
        """

        I = self.norm_onezero()
        phi = np.linspace(0, 2*np.pi, len(I))
        phi = phi[:, np.newaxis]  # Reshape for compatibility

        # HYPERPARAMETERS
        if length_scale is None:
            length_scale = (2*np.pi/len(I)) + 0.1 * 2*np.pi/len(I)
        else:
            length_scale = length_scale
        if amplitude is None:
            amplitude = 1.0
        else:
            amplitude = amplitude

        if num_samples is not None:
            I_fit, std, yerr, samples = self.GP_regression(length_scale=length_scale, 
                                                           amplitude=amplitude, 
                                                           on_pulse_bins=on_pulse_bins, 
                                                           num_samples=num_samples)
        else:
            I_fit, std, yerr = self.GP_regression(length_scale=length_scale, 
                                                  amplitude=amplitude, 
                                                  on_pulse_bins=on_pulse_bins)
        log_I = np.log(I_fit)
        phi_fit_od = np.linspace(0, 2*np.pi, len(I_fit))  # one-dimensional
        phi_fit = phi_fit_od[:, np.newaxis]

        unnormalized_knee_point, xc, yc, distances = self.get_knee_point(bins=bins, all_params=True)

        if get_error:
            knee_plus, xc_plus, yc_plus, distances_plus = self.get_knee_point(I_fit + std, bins=bins, all_params=True)
            knee_minus, xc_minus, yc_minus, distances_minus = self.get_knee_point(I_fit - std, bins=bins, all_params=True)

            widths, widths_plus, widths_minus, threshold_plus, threshold_minus = self.get_widths(get_error=True, all_info=True)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5), dpi=200)
        fig.patch.set_facecolor('white')

        # Set the axis background color to white
        # ax1.set_facecolor('white')
        # ax2.set_facecolor('white')
        # ax3.set_facecolor('white')

        # PLOT 1 - Pulse Profile and GP Fit
        ax1.scatter(phi, I, color="red", s=1, alpha=0.3, label=r"Data ($y_{err} = %.4f$)" % yerr)
        ax1.plot(phi_fit, I_fit, color="black", lw=1, zorder=10)
        ax1.plot(phi_fit, I_fit + std, color="blue", lw=1, alpha=0.5)
        ax1.plot(phi_fit, I_fit - std, color="blue", lw=1, alpha=0.5)
        ax1.fill_between(phi_fit.flatten(), I_fit - std, I_fit + std, color="steelblue", alpha=0.2, label="1$\sigma$ Confidence")
        ax1.axhline(np.exp(unnormalized_knee_point), color='red', alpha=0.5, lw=1, label='Knee point = %.6f' % np.exp(unnormalized_knee_point))
        ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
        # Plot the sample fits
        if num_samples is not None:
            for i in range(len(samples)):
                ax1.plot(phi_fit, samples[i], color="black", lw=1, alpha=0.2)

        # shading region above the knee point
        ax1.fill_between(phi_fit_od, np.max(I_fit), 0, where=I_fit > np.exp(unnormalized_knee_point), color='limegreen', alpha=0.1) 

        # Error in Widths
        
        if get_error:

            # horizontal threshold lines
            ax1.axhline(threshold_plus, color='red', alpha=0.2, lw=1)
            ax1.axhline(threshold_minus, color='red', alpha=0.2, lw=1)
            
            ax1.fill_between(phi_fit_od, np.max(I_fit), 0, where=I_fit > threshold_plus, color='limegreen', alpha=0.2)
            ax1.fill_between(phi_fit_od, np.max(I_fit), 0, where=I_fit > threshold_minus, color='limegreen', alpha=0.1)


        ax1.set_title('Pulse Profile')
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Intensity')
        ax1.set_xlim(xlim)
        ax1.legend()

        # ---------------------------------------------------------------

        # PLOT 2 - Log-Polar Profile
        ax2 = plt.subplot(132, projection='polar')
        ax2.grid(alpha=0.3)
        ax2.set_theta_zero_location('S')

        num_radial_ticks = 4  # Adjust this number based on your data
        ax2.set_yticks(np.linspace(np.min(log_I), np.max(log_I), num_radial_ticks))  # Fewer ticks

        ax2.plot(phi_fit_od, log_I, color='teal')
        ax2.plot(phi_fit_od, unnormalized_knee_point * np.ones_like(phi_fit_od), color='red', alpha=0.5, lw=1)
        
        ax2.set_title('Log-Polar Profile')
        # min non nan value
        min_log_I = np.min(log_I[np.isfinite(log_I)])
        min_shade = min_log_I - 0.1*min_log_I
        ax2.fill_between(phi_fit_od, 0, min_shade, where=log_I > unnormalized_knee_point, color='limegreen', alpha=0.1) 

        if get_error:
            ax2.plot(phi_fit_od, np.log(threshold_plus) * np.ones_like(phi_fit_od), color='red', alpha=0.2, lw=1)
            ax2.plot(phi_fit_od, np.log(threshold_minus) * np.ones_like(phi_fit_od), color='red', alpha=0.2, lw=1)
            ax2.fill_between(phi_fit_od, 0, min_shade, where=log_I > np.log(threshold_plus), color='limegreen', alpha=0.2)
            ax2.fill_between(phi_fit_od, 0, min_shade, where=log_I > np.log(threshold_minus), color='limegreen', alpha=0.1)

        # ---------------------------------------------------------------
    
        # PLOT 3 - Kneedle Algorithm    
        ax3.plot(xc, yc, color='teal', label='CDF (bins = %d)' % bins)

        if get_error:
            ax3.plot(xc_plus, yc_plus, color='teal', alpha=0.5, lw=1)
            ax3.plot(xc_minus, yc_minus, color='teal', alpha=0.5, lw=1)

            # fill_between xc_plus and xc_minus
            # Define a common x-axis range for interpolation
            common_x = np.linspace(min(xc_plus.min(), xc_minus.min()), max(xc_plus.max(), xc_minus.max()), 1000)

            from scipy.interpolate import interp1d
            # Interpolate yc_plus and yc_minus to the common x-axis
            yc_plus_interp = interp1d(xc_plus, yc_plus, bounds_error=False, fill_value="extrapolate")(common_x)
            yc_minus_interp = interp1d(xc_minus, yc_minus, bounds_error=False, fill_value="extrapolate")(common_x)

            # Fill between the interpolated values
            ax3.fill_between(common_x, yc_plus_interp, yc_minus_interp, color='teal', alpha=0.2, label='±1σ Region')

            ax3.axvline(np.log(threshold_plus), color='red', alpha=0.2, lw=1)
            ax3.axvline(np.log(threshold_minus), color='red', alpha=0.2, lw=1)

            ax3.plot(xc_plus, distances_plus, color='gray', alpha=0.2)
            ax3.plot(xc_minus, distances_minus, color='gray', alpha=0.2)

        ax3.plot(xc, distances, label='Distances', alpha=0.5, color='gray')
        ax3.axvline(unnormalized_knee_point, color='red', label='Knee point = %.2f' % unnormalized_knee_point, alpha=0.5, lw=1)

        ax3.set_xlabel('log(Intensity)')
        ax3.set_ylabel('Counts')
        ax3.set_title('Knee Detection: Kneedle Algorithm')
        ax3.legend()

        #plt.tight_layout()
        plt.suptitle('Pulse Profile and Threshold Detection', y=1.05) 
        if PSR_name is not None:
            plt.suptitle('Threshold Detection for %s' % PSR_name, y=1.05)
        plt.subplots_adjust(wspace=0.3)
        plt.show()