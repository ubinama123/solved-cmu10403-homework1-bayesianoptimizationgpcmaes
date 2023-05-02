Download Link: https://assignmentchef.com/product/solved-cmu10403-homework1-bayesian_optimizationgp_cmaes
<br>
<ul>

 <li style="list-style-type: none;"></li>

</ul>

<h1>Problem 2: Bayesian Optimization</h1>

In this section, you will implement Bayesian Optimization using Gaussian Processes and compare the average regret over time for three acquisition functions: GP-greedy, GP-UCB, and GP-Thompson.

In this section <strong>your solution will consist of a single plot</strong>, with each curve in that plot being described in a subsection below. This allows you to get partial credit if you are not able to implement all 3 acquisition functions.

We provide you with the squared exponential kernel (also known as the RBF kernel) you will use for this assignment, defined as:

In the function templates we provide, we describe the parameters:

<ol>

 <li>the parameter <em>l </em>is the scale of the RBF kernel.</li>

 <li><em>sigma f </em>is the vertical variation parameter <em><sub>f </sub></em>of the RBF kernel.</li>

 <li><em>sigma y </em>is the standard deviation of the intrinsic noise in <em>y</em>.</li>

</ol>

Feel free to experiment with values for these, but <strong>do not change the default GP arguments (arguments l, sigma f, sigma y) for the final plot</strong>. Use the following GP update equations for the posterior:

<table>

 <tbody>

  <tr>

   <td></td>

  </tr>

 </tbody>

</table>

<em>K</em><em>y </em>= <em>K </em>+ <em>y</em>2<em>I</em><em>N fy</em>⇤ = N ✓<strong>0</strong><em>,K</em><em>T<sub>y </sub>KK</em>⇤⇤<sub>⇤</sub>◆

<em>K</em>

⇤

<em>p</em>(<em>f</em><sub>⇤</sub>|<em>X</em><sub>⇤</sub><em>,X,y</em>) = N(<em>f</em><sub>⇤</sub>|<em>u</em><sub>⇤</sub><em>,</em>⌃<sub>⇤</sub>) <em>µ</em>⇤ = <em>K</em>⇤<em><sup>T</sup>K<sub>y </sub></em><sup>1</sup><em>y</em>

⌃

Review slide 193 of <a href="https://cmudeeprl.github.io/Spring202010403website/assets/lectures/s20_lecture3.pdf">https://cmudeeprl.github.io/Spring202010403website/assets/ </a><a href="https://cmudeeprl.github.io/Spring202010403website/assets/lectures/s20_lecture3.pdf">lectures/s20_lecture3.pdf</a> for more details. These equations di↵er from the ones in the slides because here we omit the o↵set in <em>µ</em><sub>⇤ </sub>since we assume zero mean.

For all questions in this section, you only need to take into account the domain defined by the python constants [<em>MIN </em><em>X,MAX X</em>]. The terms ”posterior” and ”posterior predictive” can be considered synonyms for the purposes of this question.

<ol>

 <li><strong> </strong>The first acquisition function you will implement is the simplest: GP-greedy. This function will take a training dataset, find the maximum mean value of the posterior distribution, and output the x-value corresponding to that mean. Since the GPs we are dealing with here are 1-dimensional, we recommend using grid search as an e↵ective way to estimate the maximizer.</li>

</ol>

More specifically, to do grid search on a function <em>f</em>, construct a grid of uniformly spaced points using numpy’s linspace function, evaluate the function <em>f </em>at all such points, and return the grid point with maximum value. We recommend you use at least 100 such grid points when finding the maximum.

<ol start="2">

 <li><strong> </strong>Next, implement the GP-UCB acquisition function described in class. This function is similar to the greedy one but you will need to take into account the standard deviation of each point you evaluate in the grid search. Use the following optimization process to choose the next <em>x<sub>t </sub></em>to evaluate:</li>

</ol>

<em>x<sub>t </sub></em>= argmax<em>µ</em>(<em>x</em>) + 1<em>.</em>96 (<em>x</em>)

<em>x</em>

where <em>µ</em>(<em>x</em>) represents the GP’s mean at <em>x </em>and (<em>x</em>) represents the standard deviation at <em>x</em>. Notice here that we provide a constant 1<em>.</em>96 instead of the <em><sub>t</sub></em><sup>1<em>/</em>2 </sup>described in class, which is a function of time. While having a time-varying constant is important for some applications, we choose a constant here to make convergence faster.

<ol start="3">

 <li><strong> </strong>Finally, implement Thompson sampling using the GP. This involves sampling a function from the GP and then taking the argmax of this sampled function.</li>

</ol>

Once you have completed the subsections, <strong>plot a graph of the regrets of the acquisition functions you implemented using the provided function </strong><em>bayes opt(n trials=200, n steps=50)</em>. This might take a while, so try to debug with fewer <em>n trials</em>. Finally, <strong>include a short description of the regret curve of each acquisition function and why you believe the regret behaves the way it does for that function (does it reach exactly zero regret? If it doesn’t, why not? etc.).</strong>

<h1>Problem 3: CMA-ES</h1>

For this problem, you will be working with the Cartpole (Cartpole-v0) environment from OpenAI gym. Please take a look at the implementation to learn more about the state and action spaces: <a href="https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py">https://github.com/openai/gym/blob/master/gym/envs/classic_ </a><a href="https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py">control/cartpole.py</a><a href="https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py">.</a>

For Cartpole, we have continuous states and hence we cannot use policy learning methods for finite MDPs. In class, we learned about covariance matrix adaptation evolutionary strategy (CMA-ES), a black box, gradient-free optimization method. In this problem, you will implement CMA-ES to find a policy for the Cartpole environment. There is template code provided for CMA-ES in the Colab notebook. Start by filling in the missing pieces in the CMAES class. For the CMA-ES update equations, follow the hints mentioned in the Colab notebook comments.

<ol>

 <li><strong> </strong>Run CMA-ES on Cartpole for 200 iterations (or until it reaches a reward of 200), using the provided hyperparameters. Plot the best point reward and average point reward throughout training.</li>

 <li>Run your CMA-ES algorithm with population size of 20, 50, and 100. For each of the population sizes, run CMA-ES for 200 iterations (or until it reaches a reward of 200) and plot the best point reward at each iteration. In another plot, plot the best point reward as a function of the number of weights evaluated (# of iterations ⇥ population size).</li>

 <li>(Optional) We provided most of the hyperparameters to you for this problem. However, in practice, significant time is spent tuning hyperparameters. Play around with the dimensionality of your network as well as other parameters of CMA-ES to get a feel for how the algorithm behaves.</li>

</ol>

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://www.cmu.edu/policies/">https://www.cmu.edu/policies/</a>