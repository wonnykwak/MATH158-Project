Abstract
We aim to present a new machine learning framework for golf club selection based on a player’s expected strokes to hole out conditioned on lie and distance together with the player’s personal Trackman data. This effort allows us to simulate out- comes for different clubs and aimpoints, recommending the optimal strategy to minimise expected strokes under both neutral and goal-specific loss functions tai- lored to a specific player.
1 Introduction
As world-renowned golfer Arnold Palmer once remarked, ”Golf is deceptively simple and endlessly complicated.” What seems like a basic task – moving a ball from point A to point B – quickly unravels into a series of nuanced decisions involving lie conditions, environmental variables, club selection, and shot trajectory.
In the post-COVID surge of golf’s popularity, data-driven tools have grown, but the sport still lags behind others like baseball, where analytics have transformed live decision-making. Golf’s major leap came with Strokes Gained [1] [3], introduced by Mark Broadie in 2011, but most existing tools are either retrospective: focusing on performance summaries; or grossly generalised: failing to provide player-specific insight or playing style [4].
With improvements in ML, new approaches have started to gain popularity, such as deep learning models to predict and offer the best club – based on environmental and personal shot factors [2]. We aim to incorporate this approach, shifting the emphasis from passive analysis to personalised actionable insights. We propose a ML framework that leverages player-specific data to provide rec- ommendations tailored to course conditions; individual shot tendencies; and differing risk profiles. We do this by implementing Gaussian Process Regression (GPR)[5]; player specific dispersion mod- els from Trackman data; and implementing different loss-functions. In section 2 and 3, we will cover Data and Methodology respectively, and in 4 and 5, we will discuss our results and future work.
2 Data
For this project, we chose to use two core data sets:
1. A collection of over 120, 000 strokes from Pomona-Pitzer tournament rounds, encoding initial lie type and distance to pin (yards for most lies, feet for green data), from which we were able to retrieve shots to hole out from different lies and distances. We filtered our data to within 70 yards of a reachable target.
All data was converted to yards, separated into different lies, and binned to 3-yard inver- vals to enable GPR on the aggregated averages – reducing the computational overhead of running GPR on large discrete data sets.
2. A smaller Trackman launch monitor dataset from a D3 golfer, capturing overlapping dis- tance and dispersion patterns for multiple clubs and swing lengths.
3 Methodology
After cleaning, averaging and recording observation counts for our data, we modelled the ex- pected strokes to hole out (ESHO) from within 70 yards using separate Gaussian Process Regres- sion (GPR) models for each lie type (green, fairway, rough, sand), implemented in Python with scikit-learn [6]. GPR was chosen for its non-parametric flexibility and suitability for predom- inantly smooth data. We tested both RBF and RBF+White kernels to account for potential noise and varying sample sizes for different lie types, selecting the best-performing kernel via cross-validation. Kernel hyperparameters were optimised using the original package, as full k-fold validation proved too computationally intensive for our devices.
1

To model our collected data, we created a model par-3 hole design illustrated in figure 1 using the Shapely library [7]. To resemble a realistic golf course, we included several lies: fairway, green, sand, and water. This digital canvas allowed us to overlay Trackman shot dispersion data derived from our golfer’s 7-iron, 8-iron and 9-iron shots. The metric “Carry Flat - Side” denoted the x-value and “Carry Flat - Length” was the y-value in the (x, y) coordinates. For each landing coordinate, the Shapely polygon boundaries classified the lie type and its Euclidean distance from the pin set at coordinate (0, 140) yards. With the type of lie classified, we assigned the appropriate lie- specific model to measure the ESHO of each landed shot. Then each club’s shots ESHO values were averaged, and the club with the lowest mean was recommended.
In addition to the selected club, we tested different aim points (aim angles) 1. To rotate shot disper- sions from the existing dataset, we used a rotation matrix moving the aim direction (-10° to +10°). For each varying degree, we calculated the average ESHO, identifying the optimal combination of club and aim direction.
To account for special goals, we incorporated an exponential loss function penalising shots depend- ing on wanted outcomes (eg. required birdie). Goal Loss = n1 Pni=1 ek(xi−t) denotes the loss function where k is the sharpness parameter, xi is the ESHO value for shot i, and t is the desired score (eg. 2 = birdie). We wanted to penalise scores higher than our desired score exponentially.
4 Results
Our GPR distributions exhibited the expected trends: mostly monotonic behaviour with fairway lies yielding lowest ESHO, followed by rough and then sand. It is valuable to note that the GPR revealed non-linear behaviour in bunker scenarios, with higher ESHO for shorter shots – this can easily be explained by the difficulty of short-sided bunker shots.
We successfully managed to model a toy example for our golfer, who, when going for a 140 yard pin guarded by a lake and bunker as shown below, would on average do best by choosing a full 8-iron swing aimed 1◦ (2.5 y) from table 2 left of the target with ESHO mean value: 1.5. This combination fared better when compared to half 7-iron and full 9-iron shots hit at their respective optimal aimpoints seen in 1.
  (a) GPR distributions: left, raw output, right, 2d representation centred around pin with concentric circles and 5 yard increments outlined
(b) Output of running our data, optimal strategy outlined in green in the middle: 8-iron 2.5 y left,
 ̄
ESHO = 1.5
Figure 1: Project Results
Consequently, our goal-oriented loss function reinforced this conclusion, where the 8-iron returned the lowest values when aiming to birdie (8.22) or par (0.409) the hole, outperforming the 7-iron and 9-iron seen in table 3.
5 Discussion and Future Work
Our model managed to provide interpretable, personalised club recommendations successfully. Fu- ture extensions of this work include: modeling more granular course layouts based real world; mod- eling different clubs and longer distances; considering height above sea level, wind and elevation; expanding to other players; implementing confidence bounds returned by GPR into our model; and incorporating multi-shot strategy via Markov models.
2

Appendix
Glossary of Golf Terms
Lie: The ground condition or surface where a ball rests. Common lies include:
Fairway: closely mown grass.
Rough: longer grass bordering fairway lies, thicker and more unpredictable.
Sand/bunker: obstacle or hazard filled with sand.
Green: short grass area around the hole used for putting, that is, rolling the ball into the hole.
Club: instrument used to hit golf balls. Clubs vary in loft and length, affecting distance and shot shape. Examples used in this paper include: 9-iron, 8-iron, and 7-iron.
Par: The expected number of strokes a skilled golfer should take to complete a hole. A par-3 hole expects a golfer to finish in 3 shots.
Birdie: The score that is one stroke under par on a hole. For example, if a hole has a par of 4, and a player completes it in 3 strokes, that’s a birdie.
Bogey: The score that is one stroke over par on a hole. For example, if a hole has a par of 4, and a player completes it in 5 strokes, that’s a bogey.
Dispersion: The spread of a player’s shots around their intended target, variable depending on golf club difficulty and individual golfer’s tendencies.
Aimpoint: Directional target chosen by the player, defining the centre of their shot disper- sion.
Trackman: A golf radar system that captures detailed shot data, including launch angle, ball speed, spin, dispersion and more.
ESHO: The number of strokes a player is expected to take from a given location until the ball is in the hole. Our model predicts this value.
Assumptions and Simplifications of Model
To make our project feasible and focussed given time constraints and the availability of data, we made the following assumptions:
• Flat Terrain: We ignore slope, elevation changes, and green contours in our golf hole model and when creating GPR distributions for ESHO.
• No Meteorological Factors: We assume no wind, humidity, or temperature effects, with the data from Trackman being normalised to room temperature, i.e. 21 ◦C.
• No Tree Interference: We assume open lines of play with no obstructions between the ball and target.
• Straight Ball Flights: Shot curvature is not modelled explicitly in dispersion.
• Outcome as a Function of Distance Only: We model all our shots solely by distance to hole and lie, not taking into consideration the unique intricacies that every hole in the real world may bring.
More on Gaussian Process Regression
Gaussian Process Regression, is a non-parametric Bayesian regression technique that defines a dis- tribution over infinitely many functions instead of fitting a fixed functional form (polynomial or linear model for example). This flexibility made GPR particularly appealing for our project, where we aimed to model expected strokes to hole out (ESHO) as a smooth, continuous surface based on distance and lie type, without making too many prior assumption.
The only postulation GPR makes is that nearby input values should produce similar outputs. In our case, this means a 45 yard shot from the rough should behave similarly to a 46 yard shot – a very reasonable expectation in golf. To formalise this, we selected a Radial Basis Function (RBF) kernel,
3

which encodes the idea that similarity decreases smoothly as inputs grow further apart. This allowed us to model each lie type (green, fairway, rough, sand) independently, with smooth transitions in predicted ESHO over distance.
We chose GPR because:
• In golf, ESHO changes relatively smoothly with distance, especially within a lie type, and GPR naturally captures this.
• GPR produces not just point estimates but also confidence intervals, which is especially useful in data-sparse areas (like short-sided bunker shots) where uncertainty is high. While we didn’t explicitly use these uncertainty bands in the decision-making for this version, we see strong potential for them in future iterations.
• GPR is robust to noise, and kernel selection gives us control over how smooth or flexible our predictions should be.
To choose the best kernel for each lie, we ran experiments using both a simple RBF kernel and an RBF combined with a WhiteKernel (which explicitly models observation noise). For each lie type, we fit both versions and used cross-validation to compare performance. The results were mixed - for some lies (like fairway), the RBF kernel alone worked well (probably due to the density of observations), while others (like sand) benefited from the added noise term. We allowed the library’s internal optimisation mechanism to fit the best hyperparameter selection after attempting to run k-fold cross validation for this ourselves (the data set is larger, and smaller kernel length-scale widths were giving us trouble). Below you can observe in greater detail the output of our regression.
An important advantage of modelling each lie type with its own GPR is the flexibility it gives
Figure 2: GPR prediction distribution for each lie type with binned average values.
Figure 3: GPR heatmaps for Fairway, Rough, and Sand lies. These allow us to get more of an intuition for how ESHO changes as you get away from the pin. The concentric dashed circles indicate 5 yard increments from the pin.
us in building more complex, spatially-aware simulations. For instance, the toy green shown below includes rough, sand, and green regions to a much greater granularity than the hold we experimented on in this project. Because we trained separate GPRs for each lie type, we can compute the ESHO from any location simply by identifying what surface the ball has landed on (green vs. bunker). This
  4

modular design allows us to scale our model to arbitrarily complex holes and unseen holes - as long as the landing zone is labeled, we can look up the appropriate GPR and assign an expected strokes value.
Figure 4: Left: Toy hole design showing lie types (green, rough, sand). Right: GPR based ESHO surface centered around the pin with concentric circles at 5 yard increments.
Figure 5: Shot Overlay Plot with Expected Strokes
    Result Tables:
5

 Angle (°)
-10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10
7 Iron
2.70 2.62 2.55 2.43 2.32 2.20 2.13 2.08 2.07 2.05 2.04 2.06 2.09 2.12 2.26 2.38 2.62 2.83 3.00 3.06 3.16
8 Iron
2.40 2.24 2.06 1.92 1.78 1.70 1.63 1.57 1.52 1.49 1.50 1.55 1.64 1.77 1.92 2.14 2.42 2.53 2.59 2.83 3.18
9 Iron
2.48 2.35 2.20 2.11 2.04 1.95 1.91 1.86 1.85 1.85 1.90 1.95 2.03 2.10 2.20 2.30 2.51 2.63 2.83 3.04 3.28
                      Table 1: Mean Expected Strokes
to Hole Out by Aim Angle (±10°)
 Club
7 Iron 8 Iron 9 Iron
Optimal Angle (°) and Mean
+0◦ (2.04) −1◦ (1.49) −2◦ (1.85)
    Table 2: Optimal Aim Angle for Mean Expected Strokes to Hole Out
 Club
Need Birdie
155.496
8.220
141.105
Table 3: Playoff Loss Function Values (Birdie & Par)
 7 Iron 8 Iron 9 Iron
Need Par
7.742
0.409
  Contributions
• Federica: Originated project concept, including use of GPR, TrackMan-based personalised dispersion modeling and loss-function base strategy. Collected TrackMan data and data used for GPR modeling. Cleaned GPR data, implementing GPR first in R and later in Python. Tuned kernel parameters and explored multiple cross-validation methods. Cre- ated GPR visualisations, and helped outlining the simulation framework. Contributed to literature review and co-authored presentation and final paper.
• Stephen: Cleaned and parsed TrackMan data, created toy hole and implemented the toy hole simulator using shapely. Integrated club-specific dispersion patterns into the simu- lation, and developed the pipeline to compute expected strokes to hole out (ESHO) across different aimpoints and club choices. Encoded and tested the loss functions for strategic decision-making. Contributed to the literature review and co-authored both the presentation and final paper.
6
7.025
 
References / Literature Review
[1] Broadie, M. Assessing Golfer Performance on the PGA Tour. Interfaces. 42 (2011,2)
Summary: In the revolutionary paper that would cement his reputation as the father of golf statistics, Broadie comes up with the concept of Strokes Gained (SG), which has since become the industry standard statistic to talk about player performance. He outlines the foundational issues of traditional golf statistics (“greens in regulation”, GIR; “putts per round” etc.) – the fact that they don’t account for context (missing a green by 50 yards vs 5 yards). His proposed system, SG, measures how much better or worse a player performs on a shot compared to the PGA tour average from the same location (distance and lie). He uses a statistical model parallel to our proposed ESHO to evaluate a golfer’s performance. We build on similar ideas but propose a model that takes player averages to recommend player specific strategy.
[2] Khazaeli, A. Golf Club Selection with AI-Based Game Planning. Proceedings of the Interna- tional Conference on Sports Analytics, (2025)
Summary: AI based club selection that provides the four best club options on a player’s max shot data for each club. Uses a probabilistic classification model to identify whether the club se- lected would provide a birdie, par zone, bogey zone or something worse than that. Data gathered from NCAA division 1 golfers, at a total of 1500 shots recorded. Given a specific scenario on the course, the ML model would recommend a club for that situation. Measured validity of each player through a ranking system: 1) scoring average 2) 7-iron distance 3) 7-iron approximate dispersion width 4) 7-iron approximate dispersion depth 5) Short game.
PriorRank+Pwt ∗ft Rank= PriorWeightofRank+Pwt
where wt is the weighting system and ft is the player feature value. The feature vector had 27 dimensions, taking into account shot scenario, shot selection, and shot result. A player’s calculated distance of each club was averaged by their average 7-iron shot distance and with these calculations, the four best clubs were recommended to the player based on the specific features. Focus of the paper’s approach is to weigh in the atmospheric and terrain features that impact a golf shot, not just simply shot shape and distance. Notably, by adding in more features about atmosphere and terrain lowered the out of sample cross entropy errors. The paper also incorporated a deep-learning prediction model that would predict the outcome of a shot based on highlighted features and the player’s information. Comparing with 5 different models: deep learning, gradient boosted trees, naive bayes, random forest and eXtreme gradient boosting, the paper illustrated that the deep learning model was the most accurate with 72%. Moving forward, this paper demonstrates the usefulness of incorporating machine learning into club selection and introduces the idea of integrating environmental factors into our future work as well.
[3] Mark Broadie. Every Shot Counts: Using the Revolutionary Strokes Gained Approach to Im- prove Your Golf Performance and Strategy. Gotham Books, New York, (2014)
Summary: The book preceding his revolutionary paper further democratises Strokes Gained (SG) and derives generalised statistical insights from his previous work. Retrospectively, he outlines what kinds of shots matter most in terms of general score – driving and approach. This paper further outlines the value of looking at expectation to measure performance and reassures us in the value of looking at approach shots (shots to the green). We, again, are proposing a more intricate model that returns player-specific recommendations over generalised statistical insights.
[4] Stauffer, G. & Guillot, M. Golf Strategy Optimization and the Value of Golf Skills. (2024), https://arxiv.org/abs/2309.00485
Summary: This paper develops a Markov Decision Process framework to optimise golf strat- egy using realistic shot distributions derived from ShotLink and partial TrackMan data. The authorise simulate play over a discretised 2D course layout making the process computatation- ally feasible. While they somewhat embed club tendencies using dispersion patterns, our model estimates expected strokes to hole out (ESHO) using Gaussian Process Regression (GPR), with the potential to model club-specific tendencies explicitly. Additionally, unlike their skill-based optimisation, we focus on personalising strategy through loss functions that reflect different playing styles. Their model is based on professionals’ data, while we limit ourselves to the data amateurs are able to record.
 7

[5] Rasmussen, C. E. and Williams, C. K. I. Gaussian Processes for Machine Learning. MIT Press, 2006.
Summary: This text provides the theoretical backbone for how GPR works, introducing it as a non parametric Bayesian approach to regression. It formalises the role of kernels in expressing similarity and lays out the mathematical formulation of GPR’s predictive mean and variance. We particularly looked at chapter 2 when trying to understand how GPR is performed and what our choice of kernel can do.
[6] Scikit-learn Developers. Gaussian Processes, scikit-learn documentation
Summary: Used the documentation as a reference for implementing GPR in our model .
[7] Shapely Developers. Shapely Manual, shapely.readthedocs.io
Summary: Used the Shapely manual to implementing geometric modeling of our model par-3 by creating water and sand hazards using SPolygon.
8
