\name{summary.OBsProb}
\alias{summary.OBsProb}
\title{Summary of Posterior Probabilities from Objective Bayesian Design}
\description{
    Reduced printing method for class \code{OBsProb} lists. Prints
    posterior probabilities of factors and models from Objective Bayesian procedure.}
\usage{
    \method{summary}{OBsProb}(object, nTop = 10, digits = 3, ...)
}
\arguments{
  \item{object}{list. \code{OBsProb} class list. Output list of \code{\link{OBsProb}} function.}
  \item{nTop}{integer. Number of the top ranked models to print.}
  \item{digits}{integer. Significant digits to use.}
  \item{...}{additional arguments passed to \code{summary} generic function.}
} \value{
    The function prints out the marginal factors and models posterior
    probabilities. Returns invisible list with the components:
    \item{calc}{Numeric vector with basic calculation information.}
    \item{probabilities}{Data frame with the marginal posterior probabilities.}
    \item{models}{Data frame with the models posterior probabilities.}
}

\references{ 

Box, G. E. P and R. D. Meyer (1986). "An Analysis for
Unreplicated Fractional Factorials". \emph{Technometrics.} Vol. 28. No. 1.
pp. 11--18.

Box, G. E. P and R. D. Meyer (1993). "Finding the Active Factors
in Fractionated Screening Experiments".
\emph{Journal of Quality Technology.} Vol. 25. No. 2. pp. 94--105.

Consonni, G. and Deldossi, L. (2015), "Objective Bayesian model discrimination 
in follow-up experimental designs" DOI 10.1007/s11749-015-0461-3. 
\emph{TEST}.
}

\author{Marta Nai Ruscone.}
\seealso{
    \code{\link{OBsProb}}, \code{\link{print.OBsProb}}, \code{\link{plot.OBsProb}}.
}

\examples{
library(OBsMD)
data(OBsMD.es5, package="OBsMD")
X <- as.matrix(OBsMD.es5[,1:5])
y <- OBsMD.es5[,6]
# Using for model prior probability a Beta with parameters a=1 b=1
es5.OBsProb <- OBsProb(X=X,y=y, abeta=1, bbeta=1, blk=0,mFac=5,mInt=2,nTop=32)
print(es5.OBsProb)
summary(es5.OBsProb)
}
\keyword{ design }
