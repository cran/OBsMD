\name{summary.OMD}
\alias{summary.OMD}
\title{Summary of Optimal OMD Follow-Up Experiments}
\description{
    Reduced printing method for lists of class \code{OMD}. It displays the
    best extra-runs according to the OMD criterion together with the correspondent OMD value. 
}
\usage{
    \method{summary}{OMD}(object, digits = 3, verbose=FALSE, ...)
}
\arguments{
  \item{object}{list of \code{OMD} class. Output list of \code{\link{OMD}} function.}
  \item{digits}{integer. Significant digits to use in the print out.}
  \item{verbose}{logical. If \code{TRUE}, the \code{unclass}-ed \code{object} is displayed.}
  \item{...}{additional arguments passed to \code{summary} generic function.}
}
\value{
    It prints out the marginal factors and models posterior probabilities and
    the top OMD follow-up experiments with their corresponding OMD statistic.}
    

\references{
Box, G. E. P and R. D. Meyer (1993). "Finding the Active Factors
in Fractionated Screening Experiments".
\emph{Journal of Quality Technology.} Vol. 25. No. 2. pp. 94--105.

Consonni, G. and Deldossi, L. (2015), "Objective Bayesian model discrimination 
in follow-up experimental designs" DOI 10.1007/s11749-015-0461-3. 
\emph{TEST}.

Meyer, R. D., Steinberg, D. M. and Box, G. E. P. (1996). "Follow-Up Designs
to Resolve Confounding in Multifactor Experiments (with discussion)".
\emph{Technometrics}, Vol. 38, No. 4, pp. 303--332.
}

\author{Marta Nai Ruscone.}
\seealso{\code{\link{print.OMD}} and \code{\link{OMD}}}


\examples{
library(OBsMD)
data(OBsMD.es5, package="OBsMD")
X <- as.matrix(OBsMD.es5[,1:5])
y <- OBsMD.es5[,6]
es5.OBsProb <- OBsProb(X=X,y=y,blk=0,mFac=5,mInt=2,nTop=32)
nMod <- 26
optop <- es5.OBsProb$ptop[1:nMod]
osigtop <- es5.OBsProb$sigtop[1:nMod]
onftop <- es5.OBsProb$nftop[1:nMod]
ojtop <- es5.OBsProb$jtop[1:nMod,1:max(onftop[1:nMod])]     
Xcand <- matrix(c(-1,	-1,	-1, -1,	-1,
1,	-1,	-1,	-1,	-1,
-1,	1,	-1,	-1,	-1,
1,	1,	-1,	-1,	-1,
-1,	-1,	1,	-1,	-1,
1,	-1,	1,	-1,	-1,
-1,	1,	1,	-1,	-1,
1,	1,	1,	-1,	-1,
-1,	-1,	-1,	1,	-1,
1,	-1,	-1,	1,	-1,
-1,	1,	-1,	1,	-1,
1,	1,	-1,	1,	-1,
-1,	-1,	1,	1,	-1,
1,	-1,	1,	1,	-1,
-1,	1,	1,	1,	-1,
1,	1,	1,	1,	-1,
-1,	-1,	-1,	-1,	1,
1,	-1,	-1,	-1,	1,
-1,	1,	-1,	-1,	1,
1,	1,	-1,	-1,	1,
-1,	-1,	1,	-1,	1,
1,	-1,	1,	-1,	1,
-1,	1,	1,	-1,	1,
1,	1,	1,	-1,	1,
-1,	-1,	-1,	1,	1,
1,	-1,	-1,	1,	1,
-1,	1,	-1,	1,	1,
1,	1,	-1,	1,	1,
-1,	-1,	1,	1,	1,
1,	-1,	1,	1,	1,
-1,	1,	1,	1,	1,
1,	1,	1,	1,	1
),nrow=32,ncol=5,dimnames=list(1:32,c("A","B","C","D","E")),byrow=TRUE)
p_omd <- OMD(X=X,y=y,nFac=5,nBlk=0,mInt=2,nMod=26,optop=optop,osigtop=osigtop,
onftop=onftop,ojtop=ojtop,nFoll=4,Xcand=Xcand,mIter=20,nStart=25,startDes=NULL,
top=30)
summary(p_omd)
}
\keyword{ design }