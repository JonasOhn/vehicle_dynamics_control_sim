\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{CSUniSchoolLabReport}[2022/03/21 Creodocs University/School Laboratory Report Class v4.0]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}} % Pass through any extra options specified to the base class
\ProcessOptions\relax % Process class options

\LoadClass{report} % Load the base class

%----------------------------------------------------------------------------------------
%	REQUIRED PACKAGES AND MISC CONFIGURATIONS
%----------------------------------------------------------------------------------------

\usepackage{listings}
\usepackage{enumitem} % Required for list customization
\renewcommand{\labelenumi}{\alph{enumi}.} % Change numbering in the enumerate environment by letter rather than number

\setlength{\parindent}{0pt} % Suppress paragraph indentation

\usepackage{graphicx} % Required for including images
\graphicspath{{Figures/}{./}} % Specifies where to look for included images (trailing slash required)

\usepackage{float} % Allows more precisely positioning floats e.g. \begin{figure}[H]

\usepackage{mhchem} % Package for chemical equation typesetting
\usepackage{siunitx} % Provides the \SI{}{} and \si{} commands for typesetting technical/scientific SI units correctly

\usepackage{amsmath} % Required for some math elements
\usepackage{hyperref}
\hypersetup{
	colorlinks   = true,    % Colours links instead of ugly boxes
	urlcolor     = blue,    % Colour for external hyperlinks
	linkcolor    = blue,    % Colour of internal links
	citecolor    = blue      % Colour of citations
}

%----------------------------------------------------------------------------------------
%	MARGINS
%----------------------------------------------------------------------------------------

\usepackage[
	top=3.0cm, % Top margin
	bottom=2.0cm, % Bottom margin
	left=2cm, % Left margin
	right=2cm, % Right margin
	%showframe % Uncomment to show frames around the margins for debugging purposes
]{geometry}

%----------------------------------------------------------------------------------------
%	FONTS
%----------------------------------------------------------------------------------------

\usepackage[T1]{fontenc} % Output font encoding for international characters
\usepackage[utf8]{inputenc} % Required for inputting international characters

\usepackage[scale=0.96]{XCharter} % Use the XCharter text font
% \usepackage[xcharter]{newtxmath} % Set the math font

%----------------------------------------------------------------------------------------
%	BIBLIOGRAPHY
%----------------------------------------------------------------------------------------

\usepackage[
	backend=biber, % Use the biber backend for compiling the bibliography
	citestyle=ieee, % In-text citation style
]{biblatex}
