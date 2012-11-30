use ExtUtils::MakeMaker;
use File::Copy;

if ($^V lt v5.10) {

   die("Algorithm::DecisionTree has only been tested on Perl 5.10.0.\n" .
   "Your perl version is $].\n");
}

copy("perl/MANIFEST.perl","MANIFEST");

WriteMakefile(
    NAME         => 'Algorithm::DecisionTree',
    VERSION_FROM => 'lib/Algorithm/DecisionTree.pm',
    AUTHOR       => 'Avinash Kak (kak@purdue.edu)',
    ABSTRACT     => 'A Perl module for decision-tree based classification of multidimensional data',
    clean        => {FILES => join(" ",
                                   map { "$_ */$_ */*/$_" }
                                   qw( *% *.html *.b[ac]k *.old *.orig ) )
                    },
);
