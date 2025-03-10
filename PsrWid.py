import numpy as np
import matplotlib.pyplot as plt
import george
from george import kernels

# ------------------------------------------------------------------------------

from matplotlib import rcParams
plt.style.use('default')

# Set global rcParams for consistent plots
rcParams.update({
    'figure.dpi': 200,                  # Set the resolution of the figures (dots per inch)
    'figure.figsize': (8, 6),           # Default figure size (width, height) in inches

    # # box to the plot
    # 'axes.linewidth': 0.4,              # Set the thickness of the axes spines
    # 'axes.edgecolor': 'black',          # Set the color of the axes spines
    # 'axes.spines.top': True,           # Turn off the top spine for the plot
    # 'axes.spines.right': True,         # Turn off the right spine for the plot
    

    'font.family': 'serif',             # Use a serif font family
    'font.size': 14,                    # Set the font size for text in plots
    'axes.titlesize': 16,               # Font size for axes titles
    'axes.labelsize': 14,               # Font size for axes labels


    'xtick.labelsize': 12,              # Font size for x-axis tick labels
    'ytick.labelsize': 12,              # Font size for y-axis tick labels
    'legend.fontsize': 12,              # Font size for legend text
    'lines.linewidth': 2,               # Line width for plots
    'lines.markersize': 6,              # Marker size for points

    'savefig.dpi': 300,                 # DPI for saving figures
    'savefig.format': 'png',            # Default format for saved figures

    'axes.grid': False,                  # Enable grid for axes
    'grid.alpha': 0.7,                  # Transparency of the grid
    'grid.color': 'white',               # Grid line color

    'legend.frameon': True,             # Add a frame around the legend
    'legend.framealpha': 0.8,           # Transparency for the legend frame
    'legend.fancybox': True,            # Rounded corners for the legend frame
    'legend.edgecolor': 'black',        # Edge color for the legend frame
    'legend.facecolor': 'lavender',        # Background color for the legend
})

# Optional: Set LaTeX for mathematical text (if needed and LaTeX is installed)
rcParams['text.usetex'] = False

# ------------------------------------------------------------------------------

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
            else:                                                              # on_pulse_bins are not provided - default (400, 600)
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
        
        # if I_fit[i] <= 0, then log(I_fit[i]) is undefined. So, I_fit[i] = 0 is replaced by np.min(I_fit[I_fit > 0])
        I_fit[I_fit == 0] = np.min(I_fit[I_fit > 0])

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
                   all_info=False,
                   remove_width_less_than=0.1,
                   merge_size=None,
                   widerr_plot=False) -> np.array:
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
        widths_nom = self.Widths_at_Threshold(phi_fit, I_fit, threshold, all_info=all_info)
        widths.append(widths_nom)

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

            # if widths_plus / widths_minus == 0, return empty array and exit (noisy profiles)
            if len(widths_plus) == 0 or len(widths_minus) == 0:
                return np.array([[0,0,0]]), threshold, threshold_plus, threshold_minus, std, yerr    

            # Widths +- Error
            # Here, get_widths() function is incomplete, I want to complete it to get widths+-error. with the function get_widths_at_threshold, I can find widths, and its start and end points. so i have widths plus and widths_minus now (with starting and end points). now, i want to take widths(nominal widths) start and end points and then see if the start point has trailing widths_minus start point & leading widths_plus start point. 
            # AND, if the widths end point has trailing widths_plus end point and leading widths_minus end point. These will be the errors in the estimation of start and end points of the (nominal) widths.


            # merging the widths which are close to each other.--> within 0.15 radians
            # if end of one width is within 0.15 radians of the start of the next width, merge them and keep everything else same.
            # Merge widths that are close to each other within `merge_size`.

            if remove_width_less_than is not None:
                widths_nom = widths_nom[widths_nom[:, 2] > remove_width_less_than]

            if merge_size is not None:
                merged_widths = []
                skip_next = False  # Flag to skip the next width if merged
                
                for i in range(len(widths_nom)):
                    if skip_next:  # Skip the current entry if it's already merged
                        skip_next = False
                        continue

                    if i < len(widths_nom) - 1 and widths_nom[i, 1] + merge_size >= widths_nom[i + 1, 0]:
                        # Merge the current width with the next one
                        merged_widths.append((widths_nom[i, 0], widths_nom[i + 1, 1], widths_nom[i + 1, 1] - widths_nom[i, 0]))
                        skip_next = True  # Mark the next width as already merged
                    else:
                        # No merging needed; keep the current width as is
                        merged_widths.append(tuple(widths_nom[i]))

                # if widths_nom is empty, return empty array and exit
                if len(merged_widths) == 0:
                    return np.array([[0,0,0]]), threshold, threshold_plus, threshold_minus, std, yerr
                else:
                    widths_nom = np.array(merged_widths)


            nom_start = widths_nom[:, 0]
            nom_end = widths_nom[:, 1]
            plus_start = widths_plus[:, 0]
            plus_end = widths_plus[:, 1]
            minus_start = widths_minus[:, 0]
            minus_end = widths_minus[:, 1]

            widths_with_error = []
            wid_err_plot = []           # for plotting purposes - wid:start/end, trailing/leading start/end

            for i in range(len(widths_nom)):
                
                # trailing and leading for start
                trailing_start = minus_start[minus_start < nom_start[i]].max() if any(minus_start < nom_start[i]) else nom_start[i]
                leading_start = plus_start[plus_start > nom_start[i]].min() if any(plus_start > nom_start[i]) else nom_start[i]

                # traling and leading for end
                trailing_end = plus_end[plus_end < nom_end[i]].max() if any(plus_end < nom_end[i]) else nom_end[i]
                leading_end = minus_end[minus_end > nom_end[i]].min() if any(minus_end > nom_end[i]) else nom_end[i]

                error_plus_width = leading_end - trailing_start
                error_minus_width = trailing_end - leading_start

                error_plus = np.abs(widths_nom[i, 2] - error_plus_width)
                error_minus = np.abs(widths_nom[i, 2] - error_minus_width)

                # Avoiding large errors for high SNR profiles
                # if error_plus > 5 * error_minus, then error_plus = error_minus
                # if error_minus > 5 * error_plus, then error_minus = error_plus
                if error_plus > 5 * error_minus:
                    error_plus = error_minus
                if error_minus > 5 * error_plus:
                    error_minus = error_plus

                wid_info = widths_nom[i, 2], error_plus, error_minus
                widths_with_error.append(wid_info)
                
                wid_plt_info = nom_start[i], nom_end[i], trailing_start, leading_start, trailing_end, leading_end
                wid_err_plot.append(wid_plt_info)

            widths_with_error = np.array(widths_with_error)
            wid_err_plot = np.array(wid_err_plot)

            if widerr_plot:
                return wid_err_plot
            else:
                return widths_with_error, threshold, threshold_plus, threshold_minus, std, yerr
                #return widths_nom
        
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
                     widerr_plot=False,
                     merge_size=0.1, 
                     remove_width_less_than=0.1,
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

        # if I_fit[i] <= 0, then log(I_fit[i]) is undefined. So, I_fit[i] = 0 is replaced by np.min(I_fit[I_fit > 0])
        I_fit[I_fit == 0] = np.min(I_fit[I_fit > 0])

        phi_fit_od = np.linspace(0, 2*np.pi, len(I_fit))  # one-dimensional
        phi_fit = phi_fit_od[:, np.newaxis]

        unnormalized_knee_point, xc, yc, distances = self.get_knee_point(bins=bins, all_params=True)

        if get_error:
            knee_plus, xc_plus, yc_plus, distances_plus = self.get_knee_point(I_fit + std, bins=bins, all_params=True)
            knee_minus, xc_minus, yc_minus, distances_minus = self.get_knee_point(I_fit - std, bins=bins, all_params=True)

            widths_with_error, threshold, threshold_plus, threshold_minus, std, yerr = self.get_widths(get_error=True, all_info=True)

        if get_error and widerr_plot:
            # widths_nom[i, 2], trailing_start, leading_start, trailing_end, leading_end
            widths_and_errors = self.get_widths(get_error=True, all_info=True, widerr_plot=True, merge_size=merge_size, remove_width_less_than=remove_width_less_than)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5), dpi=300)
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
        
        
        # Error in Widths
        if get_error:

            # horizontal threshold lines
            ax1.axhline(threshold_plus, color='red', alpha=0.2, lw=1)
            ax1.axhline(threshold_minus, color='red', alpha=0.2, lw=1)
            
            # shading region above the knee point
            ax1.fill_between(phi_fit_od, np.max(I_fit), 0, where=I_fit > np.exp(unnormalized_knee_point), color='limegreen', alpha=0.1)  
            ax1.fill_between(phi_fit_od, np.max(I_fit), 0, where=I_fit > threshold_plus, color='limegreen', alpha=0.2)
            ax1.fill_between(phi_fit_od, np.max(I_fit), 0, where=I_fit > threshold_minus, color='limegreen', alpha=0.1)

        else:
            ax1.fill_between(phi_fit_od, np.max(I_fit), 0, where=I_fit > np.exp(unnormalized_knee_point), color='limegreen', alpha=0.4) # changing alpha


        if get_error and widerr_plot:
            for i in range(len(widths_and_errors)):
                # filling betwwn trailing start and leading end
                ax1.fill_between([widths_and_errors[i,2], widths_and_errors[i,5]], 0, np.max(I_fit), color='limegreen', alpha=0.1)
                # filling between trailing end and leading start
                ax1.fill_between([widths_and_errors[i,4], widths_and_errors[i,3]], 0, np.max(I_fit), color='limegreen', alpha=0.1)
                # plot the nominal widths
                ax1.fill_between([widths_and_errors[i,0], widths_and_errors[i,1]], 0, np.max(I_fit), color='limegreen', alpha=0.2) 


        ax1.set_title('Pulse Profile')
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Normalized Intensity')
        ax1.set_xlim(xlim)
        ax1.legend()

        # ---------------------------------------------------------------

        # PLOT 2 - Log-Polar Profile
        ax2 = plt.subplot(132, projection='polar')
        ax2.grid(alpha=0.3)
        ax2.set_theta_zero_location('S')

        num_radial_ticks = 4  # Adjust this number based on your data
        # ax2.set_yticks(np.linspace(np.min(log_I[log_I > 0]), np.max(log_I), num_radial_ticks))

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

        else:
            ax2.fill_between(phi_fit_od, 0, min_shade, where=log_I > unnormalized_knee_point, color='limegreen', alpha=0.4)  # changing alpha

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
