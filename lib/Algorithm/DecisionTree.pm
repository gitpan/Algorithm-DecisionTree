package Algorithm::DecisionTree;

#---------------------------------------------------------------------------
# Copyright (c) 2010 Avinash Kak. All rights reserved.
# This program is free software.  You may modify and/or
# distribute it under the same terms as Perl itself.
# This copyright notice must remain attached to the file.
#
# Algorithm::DecisionTree is a pure Perl implementation for
# constructing a decision tree from training examples of
# multidimensional data and then using the tree to classify
# such data subsequently.
# ---------------------------------------------------------------------------

use 5.10.0;
use strict;
use warnings;
use Carp;

our $VERSION = '1.2';


#############################   Constructors  #######################

# Constructor for the training data generator:
sub training_data_generator { 
    my ($class, %args) = @_;
    my @params = keys %args;
    croak "\nYou have used a wrong name for a keyword argument " .
          "--- perhaps a misspelling\n" 
          if check_for_illegal_params1(@params) == 0;
    bless {
        _output_datafile             =>   $args{output_datafile} 
                                          || croak("output_datafile required"),
        _parameter_file              =>   $args{parameter_file},
        _number_of_training_samples  =>   $args{number_of_training_samples},
        _write_to_file               =>   $args{write_to_file} || 0,
        _debug1                      =>   $args{debug1} || 0,
        _debug2                      =>   $args{debug2} || 0,
        _training_sample_records     =>   {},
        _features_and_values_hash    =>   {},
        _bias_hash                   =>   {},
        _class_names                 =>   [],
        _class_priors                =>   [],
    }, $class;
}


# Constructor for the test data generator:
sub test_data_generator { 
    my ($class, %args) = @_;
    my @params = keys %args;
    croak "\nYou have used a wrong name for a keyword argument " .
          "--- perhaps a misspelling\n" 
          if check_for_illegal_params3(@params) == 0;
    bless {
        _output_test_datafile        =>   $args{output_test_datafile} 
                                  || croak("output_test_datafile required"),
        _output_class_labels_file    =>   $args{"output_class_label_file"}
                                  || croak("output_class_label_file required"),
        _parameter_file              =>   $args{parameter_file},
        _number_of_test_samples      =>   $args{number_of_test_samples},
        _write_to_file               =>   $args{write_to_file} || 0,
        _debug1                      =>   $args{debug1} || 0,
        _debug2                      =>   $args{debug2} || 0,
        _test_sample_records         =>   {},
        _features_and_values_hash    =>   {},
        _bias_hash                   =>   {},
        _class_names                 =>   [],
        _class_priors                =>   [],
    }, $class;
}


# Constructor for decision tree induction and classification with the tree:
sub new { 
    my ($class, %args) = @_;
    my @params = keys %args;
    croak "\nYou have used a wrong name for a keyword argument " .
          "--- perhaps a misspelling\n" 
          if check_for_illegal_params2(@params) == 0;
    bless {
        _training_datafile           =>   $args{training_datafile} 
                                        || croak("training_datafile required"),
        _root_node                   =>    undef,
        _debug1                      =>    $args{debug1} || 0,
        _debug2                      =>    $args{debug2} || 0,
        _training_data_hash          =>    {},
        _features_and_values_hash    =>    {},
        _samples_class_label_hash    =>    {},
        _class_names                 =>    [],
        _class_priors                =>    [],
    }, $class;
}


#################    Classify with Decision Tree  ###################

sub classify {
    my $self = shift;
    my $root_node = shift;
    my @features_and_values = @_;
    croak "Error in the names you have used for features and/or values" 
                  if ! $self->check_names_used(@features_and_values);    
    my @class_names = @{$self->{_class_names}};
    my $feature_test = $root_node->get_feature();
    my $classification = $self->recursive_descent_for_classification( 
                                      $root_node, @features_and_values );
    if ($self->{_debug1}) {
        print "\nThe classification:\n";
        foreach my $class (@class_names) {
            print "    $class with probability $classification->{$class}\n";
        }
    }
    return $classification;
}

sub recursive_descent_for_classification {
    my $self = shift;
    my $node = shift;
    my @feature_and_values = @_;
    my @class_names = @{$self->{_class_names}};
    my $feature_test_at_node = $node->get_feature();
    my $value_for_feature;
    my @remaining_features_and_values = ();
    foreach my $feature_and_value (@feature_and_values) {
        my ($feature, $value) = $feature_and_value =~ /(.+)=>(.+)/;
        if ($feature eq $feature_test_at_node) {
            $value_for_feature = $value;
        } else {
            push @remaining_features_and_values, $feature_and_value;
        }
    }
    my $feature_value_combo = "$feature_test_at_node=>$value_for_feature"
                     if defined $feature_test_at_node;
    my @children = @{$node->get_children()};
    if (@children == 0) {
        my %answer;
        my @leaf_node_class_probabilities=@{$node->get_class_probabilities()};
        foreach my $i (0..@class_names-1) {
            $answer{$class_names[$i]} = $leaf_node_class_probabilities[$i];
        }
        return \%answer;
    }
    my $answer;
    foreach my $child (@children) {
        my @branch_features_and_values = 
               @{$child->get_branch_features_and_values()};
        my $last_feature_and_value_on_branch = pop @branch_features_and_values;
        if ($last_feature_and_value_on_branch eq $feature_value_combo) {
            $answer = $self->recursive_descent_for_classification($child, 
                                    @remaining_features_and_values);
            last;
        }
    }
    return $answer;
}    

#################    Decision Tree Construction  ###################

sub construct_decision_tree_classifier {
    my $self = shift;
    my @class_names = @{$self->{_class_names}};
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my @features = keys %features_and_values_hash;
    my @class_probabilities = map {$self->prior_probability_for_class($_)} 
                                                           @class_names;
    my $entropy = $self->class_entropy_on_priors();
    my $root_node = Node->new( undef, 
                               $entropy, 
                               \@class_probabilities, []);
    $self->{_root_node} = $root_node;
    $self->recursive_descent($root_node);
    return $root_node;
}

sub recursive_descent {
    my $self = shift;
    my $node = shift;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my @features = keys %features_and_values_hash;
    my @class_names = @{$self->{_class_names}};
    my @features_and_values_on_branch = 
          @{$node->get_branch_features_and_values()};
    my ($best_feature, $best_feature_entropy) = 
         $self->best_feature_calculator(@features_and_values_on_branch);
    return if ! defined $best_feature;
    $node->set_feature($best_feature);
    if ($best_feature_entropy < $node->get_entropy()) {
        my @values_for_feature = 
              @{$features_and_values_hash{$best_feature}};
        my @feature_value_combos = map {"$best_feature=>$_"} 
                                               @values_for_feature;
        foreach my $feature_and_value (@feature_value_combos) {
            my @extended_branch_features_and_values;
            if (!@features_and_values_on_branch) {
                @extended_branch_features_and_values = ($feature_and_value);
            } else {
                @extended_branch_features_and_values =
                    @{deep_copy_array( \@features_and_values_on_branch )};
                push @extended_branch_features_and_values, $feature_and_value;
            }
            my @class_probabilities = 
               map 
               {$self->probability_for_a_class_given_sequence_of_features_and_values(
                    $_, @extended_branch_features_and_values) }
               @class_names;
            my $child_node = Node->new( undef, $best_feature_entropy,
                                       \@class_probabilities,
                                       \@extended_branch_features_and_values);
            $node->add_child_link( $child_node );
            $self->recursive_descent( $child_node );
        }
    }
}

# Say you have landed on a new node with a history of feature-value
# pairs along the path from the root to the node.  Now you need to
# decide what feature test would work the best at this new node.
sub best_feature_calculator {
    my $self = shift;
    my @features_and_values_on_branch = @_;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my @features_already_used = [];
    foreach my $feature_and_value (@features_and_values_on_branch) {    
        my ($feature, $value) = $feature_and_value =~ /(.+)=>(.+)/;
        push @features_already_used, $feature;
    }
    my @feature_tests_not_yet_used = ();
    my @all_features = keys %features_and_values_hash;
    foreach my $feature (@all_features) {
        if ( !contained_in($feature, @features_already_used) ) {
            push @feature_tests_not_yet_used, $feature;
        }
    }
    return if @feature_tests_not_yet_used == 0;
    my @array_of_entropy_values_for_different_features;
    foreach my $i (0..@feature_tests_not_yet_used-1) {
        my @values =@{$features_and_values_hash{$feature_tests_not_yet_used[$i]}};
        my $entropy_for_new_feature;
        foreach my $value (@values) {
            my $feature_and_value_string = 
                "$feature_tests_not_yet_used[$i]=>$value";
            my  @extended_features_and_values_on_branch;
            if (@features_and_values_on_branch) {
                @extended_features_and_values_on_branch =
                  @{deep_copy_array(\@features_and_values_on_branch)};
                push @extended_features_and_values_on_branch, 
                                      $feature_and_value_string;   
            } else {
                @extended_features_and_values_on_branch  =
                    ($feature_and_value_string);
            }                      
            if (!defined $entropy_for_new_feature) {
                $entropy_for_new_feature =
                 $self->class_entropy_for_a_given_sequence_of_features_values(
                     @extended_features_and_values_on_branch) 
                 *
                 $self->probability_of_a_sequence_of_features_and_values(
                     @extended_features_and_values_on_branch);
                next;
            } else {
                $entropy_for_new_feature += 
                 $self->class_entropy_for_a_given_sequence_of_features_values(
                     @extended_features_and_values_on_branch) 
                 *
                 $self->probability_of_a_sequence_of_features_and_values(
                     @extended_features_and_values_on_branch);
            }
        }
        $array_of_entropy_values_for_different_features[$i] =
                         $entropy_for_new_feature;
    }
    my ($minimum, $index) = minimum( 
        \@array_of_entropy_values_for_different_features);
    return $feature_tests_not_yet_used[$index], $minimum;
}


#################    Entropy Calculators       #####################

sub class_entropy_on_priors {
    my $self = shift;
    my @class_names = @{$self->{_class_names}};
#    my @priors = map {$self->prior_probability_for_class($_)} @class_names;
    my $entropy;
    foreach my $class (@class_names) {
        my $prob = $self->prior_probability_for_class($class);
        my $log_prob = log($prob) / log(2) if ($prob >= 0.0001) && ($prob <= 0.999) ;
        $log_prob = 0 if $prob < 0.0001;           # since X.log(X)->0 as X->0
        $log_prob = 0 if $prob > 0.999;            # since log(1) = 0
        if (!defined $entropy) {
            $entropy = -1.0 * $prob * $log_prob; 
            next;
        }
        $entropy += -1.0 * $prob * $log_prob;
    }
    return $entropy;
}

sub class_entropy_for_a_given_sequence_of_features_values {
    my $self = shift;
    my @array_of_features_and_values = @_;
    my @class_names = @{$self->{_class_names}};
    my $entropy;
    foreach my $class (@class_names) {
        my $prob = 
         $self->probability_for_a_class_given_sequence_of_features_and_values(
             $class, @array_of_features_and_values);
        my $log_prob = log($prob) / log(2) 
                  if ($prob >= 0.0001) && ($prob <= 0.999) ;
        $log_prob = 0 if $prob < 0.0001;           # since X.log(X)->0 as X->0
        $log_prob = 0 if $prob > 0.999;            # since log(1) = 0
        if (!defined $entropy) {
            $entropy = -1.0 * $prob * $log_prob; 
            next;
        }
        $entropy += -1.0 * $prob * $log_prob;
    }
    return $entropy;
}

sub class_entropy_for_a_given_feature {
    my $self = shift;
    my $feature = shift;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my @values = @{$features_and_values_hash{$feature}};
    my $entropy;
    foreach my $value (@values) {
        if (!defined $entropy) {
            $entropy = 
                $self->class_entropy_for_a_given_feature_and_given_value(
                                                         $feature,$value)
                *
                $self->probability_for_feature_value($feature,$value);
            next;
        }
        $entropy += 
            $self->class_entropy_for_a_given_feature_and_given_value(
                                                        $feature,$value)
            *
            $self->probability_for_feature_value($feature,$value);
    }
    return $entropy;
}

sub class_entropy_for_a_given_feature_and_given_value {
    my $self = shift;
    my $feature = shift;
    my $value = shift;
    my @class_names = @{$self->{_class_names}};
    my $entropy;
    foreach my $class (@class_names) {
        my $prob = 
            $self->probability_for_a_class_given_feature_value($class,
                                                         $feature,$value);
        my $log_prob = (log($prob) / log(2)) 
                    if ($prob >= 0.0001) && ($prob <= 0.999) ;
        $log_prob = 0 if $prob < 0.0001;           # since X.log(X)->0 as X->0
        $log_prob = 0 if $prob > 0.999;            # since log(1) = 0
        if (!defined $entropy ) {
            $entropy = -1.0 * $prob * $log_prob;
            next;
        }
        $entropy += - ($prob * $log_prob);
    }
    return $entropy;
}


#################    Probability Calculators   ######################

# args order: $class_name, @array_of_features_and_values
sub probability_for_a_class_given_sequence_of_features_and_values {
    my $self = shift;
    my $class_wanted = shift;
    my @class_names = @{$self->{_class_names}};
    my @array_of_features_and_values = @_;
    my @array_of_class_probabilities = (0) x @class_names;
    foreach my $i (0..@class_names-1) {
        my $prob = 
          $self->probability_for_sequence_of_features_and_values_given_class(
                                             $class_names[$i],
                                             @array_of_features_and_values);
        $array_of_class_probabilities[$i] = 0 if $prob == 0;
        next if $prob == 0;
        # If the above prob is not zero, the following prob should also not
        # be zero:
        my $prob_of_feature_sequence = 
            $self->probability_of_a_sequence_of_features_and_values(
                                             @array_of_features_and_values);
        $array_of_class_probabilities[$i] = 
              $prob * $self->prior_probability_for_class($class_names[$i])
             / $prob_of_feature_sequence;
    }
    # The following normalization is dictated by the fact that when the
    # database is small, the answer returned by this function may not
    # add up to 1 when summed over the different classes.
    my $sum_probability = 0;
    map {$sum_probability += $_} @array_of_class_probabilities;
    @array_of_class_probabilities = map {$_ / $sum_probability} 
                                        @array_of_class_probabilities;
    my $index = get_index_at_value( $class_wanted, \@class_names );
    return $array_of_class_probabilities[$index];
}

# arg order:  classname, array of feature=>value pairs
sub probability_for_sequence_of_features_and_values_given_class {
    my $self = shift;
    my $class = shift;
    my @array_of_features_and_values = @_;
    my $probability;
    foreach my $feature_and_value (@array_of_features_and_values) {
        my ($feature, $value) = $feature_and_value =~/(.+)=>(.+)/;
        if (!defined $probability) {
            $probability = $self->probability_for_feature_value_given_class(
                                             $feature, $value, $class);
            next;
        } else {
            $probability *= $self->probability_for_feature_value_given_class(
                                       $feature, $value, $class);
        }
    }
    return $probability;
}

sub probability_of_a_sequence_of_features_and_values {
    my $self = shift;
    my @array_of_features_and_values = @_;
    my $probability;
    foreach my $feature_and_value (@array_of_features_and_values) {    
        my ($feature, $value) = $feature_and_value =~/(.+)=>(.+)/;
        if (!defined $probability) {
            $probability = 
               $self->probability_for_feature_value($feature, $value);
            next;
        } else {
            $probability *= 
                  $self->probability_for_feature_value($feature, $value);
        }
    }
    return $probability;
}

# argument order:  class_name, feature_name, feature_value
sub probability_for_a_class_given_feature_value {
    my $self = shift;
    my $class = shift;
    my $feature = shift;
    my $value = shift;
    my $prob = 
      $self->probability_for_feature_value_given_class($feature,$value,$class);
    my $answer = ($prob * $self->prior_probability_for_class($class)) 
                 /
                 $self->probability_for_feature_value($feature,$value);
    return $answer;
}

sub probability_for_feature_value {
    my $self = shift;
    my $feature = shift;
    my $value = shift;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my %training_data_hash = %{$self->{_training_data_hash}};
    my @values_for_feature = @{$features_and_values_hash{$feature}};
    @values_for_feature = map {"$feature=>" . $_} @values_for_feature;
    my @value_counts = (0) x @values_for_feature;
    foreach my $sample (sort {sample_index($a) <=> sample_index($b)}
                                              keys %training_data_hash) {
        my @features_and_values = @{$training_data_hash{$sample}};
        foreach my $i (0..@values_for_feature-1) {
            foreach my $current_value (@features_and_values) {
                $value_counts[$i]++ 
                  if $values_for_feature[$i] eq $current_value;
            }
        }
    }
    my $total_count = 0;
    my $answer;
    foreach my $i (0..@values_for_feature-1) {    
        $answer = $value_counts[$i]
              if "$feature=>$value" eq $values_for_feature[$i];
        $total_count += $value_counts[$i];
    }
    return 0 if !defined $answer;
    return $answer / (1.0 * $total_count);
}
    
# argument order:  feature_name, feature_value, class_name
sub probability_for_feature_value_given_class {
    my $self = shift;
    my $feature = shift;
    my $feature_value = shift;
    my $class = shift;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my %samples_class_label_hash = %{$self->{_samples_class_label_hash}};
    my %training_data_hash = %{$self->{_training_data_hash}};
    my @samples_for_class;
    foreach my $sample_name (keys %samples_class_label_hash) {
        push @samples_for_class, $sample_name 
            if $samples_class_label_hash{$sample_name} eq $class;
    }
    my @values_for_feature = @{$features_and_values_hash{$feature}};
    @values_for_feature = map {"$feature=>" . $_} @values_for_feature;
    my @value_counts = (0) x @values_for_feature;
    foreach my $sample (@samples_for_class) {
        my @features_and_values = @{$training_data_hash{$sample}};
        foreach my $i (0..@values_for_feature-1) {
            foreach my $current_value (@features_and_values) {
                $value_counts[$i]++ 
                  if $values_for_feature[$i] eq $current_value;
            }
        }
    }
    my $total_count = 0;
    my $answer;
    foreach my $i (0..@values_for_feature-1) {    
        $answer = $value_counts[$i] 
            if "$feature=>$feature_value" eq $values_for_feature[$i];
        $total_count += $value_counts[$i];
    }
    return 0 if !defined $answer;
    return $answer / (1.0 * $total_count);
}

sub prior_probability_for_class {
    my $self = shift;
    my $class = shift;
    my %samples_class_label_hash = %{$self->{_samples_class_label_hash}};
    my $total_num_of_samples = keys %samples_class_label_hash;
    my $count = 0;
    my @values = values %samples_class_label_hash;
    my @trues = grep {$_ eq $class} @values;
    return (1.0 * @trues) / $total_num_of_samples; 
}

###################  Read Training Data From File  ###################

sub get_training_data {
    my $self = shift;
    my @feature_names;
    my %features_and_values_hash;
    my %samples_class_label_hash;
    my %training_data_hash;
    my $training_data_file = $self->{_training_datafile};
    my $recording_features_flag = 0;
    my $recording_training_data = 0;
    my @table_header;
    my %column_label_hash;
    open INPUT, $training_data_file
                || "unable to open training data file: $!";
    my @all_data;
    my $all_data_as_string;
    my %training_data_records;

    while (<INPUT>) {
        chomp;
        next if /^[\s=#]*$/;
        if ( (/^class/i) && !$recording_training_data 
                         && !$recording_features_flag ) {
            $_ =~ /^\s*class names:\s*([ \S]+)\s*/i;
            my @class_names = split /\s+/, $1;
            $self->{_class_names} = \@class_names;
            next;
        } elsif (/^feature names and their values/i) {
            $recording_features_flag = 1;
            next;
        } elsif (/^training data/i) {
            $recording_training_data = 1;
            $recording_features_flag = 0;
            next;
        } elsif ( !$recording_training_data && $recording_features_flag ) {
            my ($feature_name, $value_string) = $_ =~ /^\s*(\S+)\s*=>\s*(.+)/i;
            $features_and_values_hash{$feature_name} = [];
            my @values = split /\s+/, $value_string;
            @values = grep $_, @values;
            push @{$features_and_values_hash{$feature_name}}, @values;
            push @feature_names, $feature_name;
        } elsif ($recording_training_data) {
            if (@table_header == 0) {
                @table_header = split;
                foreach my $i (2..@table_header-1) {
                    $column_label_hash{$i} = $table_header[$i];
                }
                my @temp = keys %column_label_hash;
                # print "keys of column_label_hash: @temp \n";
                next;
            }
            my @record = split /\s+/;
            $samples_class_label_hash{$record[0]} = $record[1];
            $training_data_hash{$record[0]} = [];
            foreach my $i (2..@record-1) {
                push @{$training_data_hash{$record[0]}}, 
                                "$column_label_hash{$i}" . "=>" . $record[$i];
            }
        }
    }
    $self->{_feature_names} = \@feature_names;
    $self->{_features_and_values_hash} = \%features_and_values_hash;
    $self->{_samples_class_label_hash} = \%samples_class_label_hash;
    $self->{_training_data_hash} = \%training_data_hash;
}    

sub show_training_data {
    my $self = shift;
    my @class_names = @{$self->{_class_names}};
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my %samples_class_label_hash = %{$self->{_samples_class_label_hash}};
    my %training_data_hash = %{$self->{_training_data_hash}};
    print "\n\nClass Names: @class_names\n";
    print "\n\nFeatures and Their Possible Values:\n\n";
    while ( my ($k, $v) = each %features_and_values_hash ) {
        print "$k --->  @{$features_and_values_hash{$k}}\n";
    }
    print "\n\nSamples vs. Class Labels:\n\n";
    foreach my $kee (sort {sample_index($a) <=> sample_index($b)} 
                                      keys %samples_class_label_hash) {
        print "$kee =>  $samples_class_label_hash{$kee}\n";
    }
    print "\n\nTraining Samples:\n\n";
    foreach my $kee (sort {sample_index($a) <=> sample_index($b)} 
                                      keys %training_data_hash) {
        print "$kee =>  @{$training_data_hash{$kee}}\n";
    }
}    

sub get_class_names {
    my $self = shift;
    return @{$self->{_class_names}}
}


###################  For Generating Your Own Training Data  ###############

sub read_parameter_file {
    my $self = shift;
    my $debug = $self->{_debug1};
    my $debug2 = $self->{_debug2};
    my $write_to_file = $self->{_write_to_file};
    my $number_of_training_samples = $self->{_number_of_training_samples};
    my $input_parameter_file = $self->{_parameter_file};
    croak "Forgot to supply parameter file" if ! defined $input_parameter_file;
    my $output_file = $self->{_output_datafile};
    
    my @all_params;
    my $param_string;
    open INPUT, $input_parameter_file
                || "unable to open parameter file: $!";
    @all_params = <INPUT>;
    @all_params = grep { $_ !~ /^[ ]*#/ } @all_params;
    chomp @all_params;
    $param_string = join ' ', @all_params;
    
    my ($class_names, $class_priors, $rest_param) = 
              $param_string =~ /^\s*class names:(.*?)\s*class priors:(.*?)(feature: .*)/;
    my @class_names = grep $_, split /\s+/, $1;
    push @{$self->{_class_names}}, @class_names;
    my @class_priors = grep $_, split /\s+/, $2;
    push @{$self->{_class_priors}}, @class_priors;    
    my ($feature_string, $bias_string) = $rest_param =~ /(feature:.*?) (bias:.*)/;
    my %features_and_values_hash;
    my @features = split /(feature[:])/, $feature_string;
    @features = grep $_, @features;
    foreach my $item (@features) {
        next if $item =~ /feature/;
        my @splits = split / /, $item;
        @splits = grep $_, @splits;
        foreach my $i (0..@splits-1) {
            if ($i == 0) {
                $features_and_values_hash{$splits[0]} = [];
            } else {
                next if $splits[$i] =~ /values/;
                push @{$features_and_values_hash{$splits[0]}}, $splits[$i];
            }
        }
    }
    $self->{_features_and_values_hash} = \%features_and_values_hash;
    my %bias_hash = %{$self->{_bias_hash}};
    my @biases = split /(bias[:]\s*class[:])/, $bias_string;
    @biases = grep $_, @biases;
    foreach my $item (@biases) {
        next if $item =~ /bias/;
        my @splits = split /\s+/, $item;
        @splits = grep $_, @splits;
        my $feature_name;
        foreach my $i (0..@splits-1) {
            if ($i == 0) {
                $bias_hash{$splits[0]} = {};
            } elsif ($splits[$i] =~ /(^.+)[:]$/) {
                $feature_name = $1;
                $bias_hash{$splits[0]}->{$feature_name} = [];
            } else {
                next if !defined $feature_name;
                push @{$bias_hash{$splits[0]}->{$feature_name}}, $splits[$i]
                        if defined $feature_name;
            }
        }
    }
    $self->{_bias_hash} = \%bias_hash;
    if ($debug) {
        print "\n\nClass names: @class_names\n";
        my $num_of_classes = @class_names;
        print "Class priors: @class_priors\n";
        print "Number of classes: $num_of_classes\n";
        print "\nHere are the features and their possible values:\n";
        while ( my ($k, $v) = each %features_and_values_hash ) {
            print "$k ===>  @$v\n";
        }
        print "\nHere is the biasing for each class:\n";
        while ( my ($k, $v) = each %bias_hash ) {
            print "$k:\n";
            while ( my ($k1, $v1) = each %$v ) {
                print "       $k1 ===>  @$v1\n";
            }
        }
    }
}

sub gen_training_data {
    my $self = shift;
    my @class_names = @{$self->{_class_names}};
    my @class_priors = @{$self->{_class_priors}};
    my %training_sample_records;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my %bias_hash  = %{$self->{_bias_hash}};
    my $how_many_training_samples = $self->{_number_of_training_samples};
    open FILEHANDLE, ">$self->{_output_datafile}" 
                        or die "Unable to open file: $!";
    my %class_priors_to_unit_interval_map;
    my $accumulated_interval = 0;
    foreach my $i (0..@class_names-1) {
        $class_priors_to_unit_interval_map{$class_names[$i]} 
         = [$accumulated_interval, $accumulated_interval + $class_priors[$i]];
        $accumulated_interval += $class_priors[$i];
    }
    if ($self->{_debug1}) {
        print "Mapping of class priors to unit interval: \n";
        while ( my ($k, $v) = each %class_priors_to_unit_interval_map ) {
            print "$k =>  @$v\n";
        }
        print "\n\n";
    }
    my $ele_index = 0;
    while ($ele_index < $how_many_training_samples) {
        my $sample_name = "sample" . "_$ele_index";
        $training_sample_records{$sample_name} = [];
        # Generate class label for this training sample:                
        my $roll_the_dice = rand(1.0);
        my $class_label;
        foreach my $class_name (keys %class_priors_to_unit_interval_map ) {
            my $v = $class_priors_to_unit_interval_map{$class_name};
            if ( ($roll_the_dice >= $v->[0]) && ($roll_the_dice <= $v->[1]) ) {
                push @{$training_sample_records{$sample_name}}, 
                                    "class=" . $class_name;
                $class_label = $class_name;
                last;
            }
        }
        foreach my $feature (keys %features_and_values_hash) {
            my @values = @{$features_and_values_hash{$feature}};
            my $bias_string = $bias_hash{$class_label}->{$feature}->[0];
            my $no_bias = 1.0 / @values;
            $bias_string = "$values[0]" . "=$no_bias" if !defined $bias_string;
            my %value_priors_to_unit_interval_map;
            my @splits = split /\s*=\s*/, $bias_string;
            my $chosen_for_bias_value = $splits[0];
            my $chosen_bias = $splits[1];
            my $remaining_bias = 1 - $chosen_bias;
            my $remaining_portion_bias = $remaining_bias / (@values -1);
            @splits = grep $_, @splits;
            my $accumulated = 0;
            foreach my $i (0..@values-1) {
                if ($values[$i] eq $chosen_for_bias_value) {
                    $value_priors_to_unit_interval_map{$values[$i]} 
                        = [$accumulated, $accumulated + $chosen_bias];
                    $accumulated += $chosen_bias;
                } else {
                    $value_priors_to_unit_interval_map{$values[$i]} 
                      = [$accumulated, $accumulated + $remaining_portion_bias];
                    $accumulated += $remaining_portion_bias;           
                }
            }
            my $roll_the_dice = rand(1.0);
            my $value_label;
            foreach my $value_name (keys %value_priors_to_unit_interval_map ) {
                my $v = $value_priors_to_unit_interval_map{$value_name};
                if ( ($roll_the_dice >= $v->[0]) 
                             && ($roll_the_dice <= $v->[1]) ) {
                    push @{$training_sample_records{$sample_name}}, 
                                            $feature . "=" . $value_name;
                    $value_label = $value_name;
                    last;
                }
            }
            if ($self->{_debug2}) {
                print "mapping feature value priors for '$feature' " .
                                          "to unit interval: \n";
                while ( my ($k, $v) = 
                        each %value_priors_to_unit_interval_map ) {
                    print "$k =>  @$v\n";
                }
                print "\n\n";
            }
        }
        $ele_index++;
    }
    $self->{_training_sample_records} = \%training_sample_records;
    if ($self->{_debug2}) {
        print "\n\nPRINTING TRAINING RECORDS:\n\n";
        foreach my $kee (sort {sample_index($a) <=> sample_index($b)} 
                                         keys %training_sample_records) {
            print "$kee =>  @{$training_sample_records{$kee}}\n\n";
        }
    }
}

sub write_training_data_to_file {
    my $self = shift;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my @class_names = @{$self->{_class_names}};
    my $output_file = $self->{_output_datafile};
    my %training_sample_records = %{$self->{_training_sample_records}};
    print "\n\nDISPLAYING TRAINING RECORDS:\n\n" if $self->{_debug1};
    open FILEHANDLE, ">$output_file";
    print FILEHANDLE "Class names: @class_names\n\n" 
                                if $self->{_write_to_file};
    print FILEHANDLE "Feature names and their values:\n" 
                                if $self->{_write_to_file};
    my @features = keys %features_and_values_hash;
    die "You probably forgot to call gen_training_data() before " .
            "calling write_training_data_to_file()()" if @features == 0;
    my %feature_name_indices;
    foreach my $i (0..@features-1) {
        $feature_name_indices{$features[$i]} = $i + 2;
        print FILEHANDLE "    $features[$i] => @{$features_and_values_hash{$features[$i]}}\n" if $self->{_write_to_file};
    }
    print FILEHANDLE "\n\nTraining Data:\n\n" if $self->{_write_to_file};
    my $num_of_columns = @features + 2;
    my $field_width = '@' . "<" x $self->find_longest_feature_or_value();
    my $fmt = "$field_width  " x $num_of_columns;
    formline( $fmt, "sample", "class", @features );
    use English;
    print $ACCUMULATOR, "\n" if $self->{_debug1};
    print FILEHANDLE "\n\n" if $self->{_write_to_file};
    print FILEHANDLE $ACCUMULATOR, "\n" if $self->{_write_to_file};
    $ACCUMULATOR = "";
    print "=" x length($fmt) . "\n\n" if $self->{_debug1};
    print FILEHANDLE "=" x length($fmt) . "\n\n" if $self->{_write_to_file};

    foreach my $kee (sort {sample_index($a) <=> sample_index($b)} 
                                     keys %training_sample_records) {
        my @record = @{$training_sample_records{$kee}};
        my @args_for_formline;
        $args_for_formline[0] = $kee;
        foreach my $item (@record) {
            $item =~ /(.+)=(.+)/;
            my ($item_name, $item_value) = ($1, $2);
            if ($item_name eq "class") {
                $args_for_formline[1] = $item_value;
            } else {
                $args_for_formline[ $feature_name_indices{$item_name} ] 
                                                 = $item_value;
            }
        }
        formline( $fmt, @args_for_formline );
        print $ACCUMULATOR, "\n" if $self->{_debug1};
        print FILEHANDLE $ACCUMULATOR, "\n" if $self->{_write_to_file};
        $ACCUMULATOR = "";
    }
    close FILEHANDLE;
}

#######################   For Generating Test Data   ####################

# Although the following method could be combined with 
# the gen_training_data() method, I have kept them separate
# to make it easier to generate test data whose stats may not
# be identical to that of the training data:
sub gen_test_data {
    my $self = shift;
    my @class_names = @{$self->{_class_names}};
    my @class_priors = @{$self->{_class_priors}};
    my %test_sample_records;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my %bias_hash  = %{$self->{_bias_hash}};
    my $how_many_test_samples = $self->{_number_of_test_samples};
    my $file_for_class_labels = $self->{_output_class_labels_file};
    open FILEHANDLE, ">$file_for_class_labels" 
       or die "Unable to open file for writing class labels for test data: $!";
    my %class_priors_to_unit_interval_map;
    my $accumulated_interval = 0;
    foreach my $i (0..@class_names-1) {
        $class_priors_to_unit_interval_map{$class_names[$i]} 
         = [$accumulated_interval, $accumulated_interval + $class_priors[$i]];
        $accumulated_interval += $class_priors[$i];
    }
    if ($self->{_debug1}) {
        print "Mapping of class priors to unit interval: \n";
        while ( my ($k, $v) = each %class_priors_to_unit_interval_map ) {
            print "$k =>  @$v\n";
        }
        print "\n\n";
    }
    my $ele_index = 0;
    while ($ele_index < $how_many_test_samples) {
        my $sample_name = "sample" . "_$ele_index";
        $test_sample_records{$sample_name} = [];
        # Generate class label for this training sample:                
        my $roll_the_dice = rand(1.0);
        my $class_label;
        foreach my $class_name (keys %class_priors_to_unit_interval_map ) {
            my $v = $class_priors_to_unit_interval_map{$class_name};
            if ( ($roll_the_dice >= $v->[0]) && ($roll_the_dice <= $v->[1]) ) {
                print FILEHANDLE "$sample_name    $class_name\n";
                $class_label = $class_name;
                last;
            }
        }
        foreach my $feature (keys %features_and_values_hash) {
            my @values = @{$features_and_values_hash{$feature}};
            my $bias_string = $bias_hash{$class_label}->{$feature}->[0];
            my $no_bias = 1.0 / @values;
            $bias_string = "$values[0]" . "=$no_bias" if !defined $bias_string;
            my %value_priors_to_unit_interval_map;
            my @splits = split /\s*=\s*/, $bias_string;
            my $chosen_for_bias_value = $splits[0];
            my $chosen_bias = $splits[1];
            my $remaining_bias = 1 - $chosen_bias;
            my $remaining_portion_bias = $remaining_bias / (@values -1);
            @splits = grep $_, @splits;
            my $accumulated = 0;
            foreach my $i (0..@values-1) {
                if ($values[$i] eq $chosen_for_bias_value) {
                    $value_priors_to_unit_interval_map{$values[$i]} 
                        = [$accumulated, $accumulated + $chosen_bias];
                    $accumulated += $chosen_bias;
                } else {
                    $value_priors_to_unit_interval_map{$values[$i]} 
                      = [$accumulated, $accumulated + $remaining_portion_bias];
                    $accumulated += $remaining_portion_bias;           
                }
            }
            my $roll_the_dice = rand(1.0);
            my $value_label;
            foreach my $value_name (keys %value_priors_to_unit_interval_map ) {
                my $v = $value_priors_to_unit_interval_map{$value_name};
                if ( ($roll_the_dice >= $v->[0]) 
                             && ($roll_the_dice <= $v->[1]) ) {
                    push @{$test_sample_records{$sample_name}}, 
                                            $feature . "=" . $value_name;
                    $value_label = $value_name;
                    last;
                }
            }
            if ($self->{_debug2}) {
                print "mapping feature value priors for '$feature' " .
                                          "to unit interval: \n";
                while ( my ($k, $v) = 
                        each %value_priors_to_unit_interval_map ) {
                    print "$k =>  @$v\n";
                }
                print "\n\n";
            }
        }
        $ele_index++;
    }
    $self->{_test_sample_records} = \%test_sample_records;
    close FILEHANDLE;
    if ($self->{_debug2}) {
        print "\n\nPRINTING TEST RECORDS:\n\n";
        foreach my $kee (sort {sample_index($a) <=> sample_index($b)} 
                                         keys %test_sample_records) {
            print "$kee =>  @{$test_sample_records{$kee}}\n\n";
        }
    }
}

sub write_test_data_to_file {
    my $self = shift;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my @class_names = @{$self->{_class_names}};
    my $output_file = $self->{_output_test_datafile};
    my %test_sample_records = %{$self->{_test_sample_records}};
    print "\n\nDISPLAYING TEST RECORDS:\n\n" if $self->{_debug1};
    open FILEHANDLE, ">$output_file";
    my @features = keys %features_and_values_hash;
    die "You probably forgot to read the parameter file before calling " .
                 "write_test_data_to_file()"   if @features == 0;
    print "Feature Order For Data:   @features\n\n\n" if $self->{_debug1};
    print FILEHANDLE "Feature Order For Data:   @features\n\n\n";
    my %feature_name_indices;
    foreach my $i (0..@features-1) {
        $feature_name_indices{$features[$i]} = $i + 1;
    }
    my $num_of_columns = @features + 1;
    my $field_width = '@' . "<" x $self->find_longest_feature_or_value();
    use English;
    my $fmt = "$field_width  " x $num_of_columns;
    foreach my $kee (sort {sample_index($a) <=> sample_index($b)} 
                                     keys %test_sample_records) {
        my @record = @{$test_sample_records{$kee}};
        my @args_for_formline;
        $args_for_formline[0] = $kee;
        foreach my $item (@record) {
            $item =~ /(.+)=(.+)/;
            my ($item_name, $item_value) = ($1, $2);
            $args_for_formline[ $feature_name_indices{$item_name} ] 
                                                 = $item_value;
        }
        formline( $fmt, @args_for_formline );
        print $ACCUMULATOR, "\n" if $self->{_debug1};
        print FILEHANDLE $ACCUMULATOR, "\n" if $self->{_write_to_file};
        $ACCUMULATOR = "";
    }
    close FILEHANDLE;
}

sub find_longest_feature_or_value {
    my $self = shift;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my $max_length;
    foreach my $feature (keys %features_and_values_hash) {
        $max_length = length $feature if ! defined $max_length; 
        $max_length = length( $feature ) if length($feature) > $max_length;
        my @values = @{$features_and_values_hash{$feature}};
        foreach my $value (@values) {
            $max_length = length( $value ) if length($value) > $max_length;
        }
    }
    return $max_length;
}

sub check_names_used {
    my $self = shift;
    my @features_and_values_test_data = @_;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my @legal_feature_names = keys %features_and_values_hash;
    foreach my $feature_and_value (@features_and_values_test_data) {
        my ($feature, $value) = $feature_and_value =~ /(.+)=>(.+)/;
        croak "Your test data has formatting error" 
            if !defined($feature) || !defined($value);
        return 0 if ! contained_in($feature, @legal_feature_names);
        my @legal_values = @{$features_and_values_hash{$feature}};
        return 0 if ! contained_in($value, @legal_values);
    }
    return 1;
}


###########################  Utility Routines  #####################

# returns the array index that contains a specified STRING value:
# meant only for array of strings
sub get_index_at_value {
    my $value = shift;
    my @array = @{shift @_};
    foreach my $i (0..@array-1) {
        return $i if $value eq $array[$i];
    }
}

sub sample_index {
    my $arg = shift;
    $arg =~ /_(.+)$/;
    return $1;
}    

# Returns the minimum value and its positional index in an array
sub minimum {
    my $arr = shift;
    my $min;
    my $index;
    foreach my $i (0..@{$arr}-1) {
        if ( (!defined $min) || ($arr->[$i] < $min) ) {
            $index = $i;
            $min = $arr->[$i];
        }
    }
    return ($min, $index);
}

# checks whether an element is in an array:
sub contained_in {
    my $ele = shift;
    my @array = @_;
    my $count = 0;
    map {$count++ if $ele eq $_} @array;
    return $count;
}

# Meant only for an array of strings (no nesting):
sub deep_copy_array {
    my $ref_in = shift;
    my $ref_out;
    foreach my $i (0..@{$ref_in}-1) {
        $ref_out->[$i] = $ref_in->[$i];
    }
    return $ref_out;
}

sub check_for_illegal_params1 {
    my @params = @_;
    my @legal_params = qw / output_datafile
                            write_to_file
                            number_of_training_samples
                            parameter_file
                            debug1
                            debug2
                          /;
    my $found_match_flag;
    foreach my $param (@params) {
        foreach my $legal (@legal_params) {
            $found_match_flag = 0;
            if ($param eq $legal) {
                $found_match_flag = 1;
                last;
            }
        }
        last if $found_match_flag == 0;
    }
    return $found_match_flag;
}

sub check_for_illegal_params2 {
    my @params = @_;
    my @legal_params = qw / training_datafile
                            debug1
                            debug2
                          /;
    my $found_match_flag;
    foreach my $param (@params) {
        foreach my $legal (@legal_params) {
            $found_match_flag = 0;
            if ($param eq $legal) {
                $found_match_flag = 1;
                last;
            }
        }
        last if $found_match_flag == 0;
    }
    return $found_match_flag;
}

sub check_for_illegal_params3 {
    my @params = @_;
    my @legal_params = qw / output_test_datafile
                            output_class_label_file
                            parameter_file
                            number_of_test_samples
                            write_to_file
                            debug1
                            debug2
                          /;
    my $found_match_flag;
    foreach my $param (@params) {
        foreach my $legal (@legal_params) {
            $found_match_flag = 0;
            if ($param eq $legal) {
                $found_match_flag = 1;
                last;
            }
        }
        last if $found_match_flag == 0;
    }
    return $found_match_flag;
}

#######################  Class Node  ###########################

# The nodes of the decision tree are instances of this class:

package Node;

use strict;                                                         
use Carp;

# $feature is the feature test at the current node.
# $branch_features_and_values is an anonymous array holding
# the feature names and corresponding values on the path
# from the root to the current node:
sub new {                                                           
    my ($class, $feature, $entropy, $class_probabilities, $branch_features_and_values) = @_; 
    bless {                                                         
        _feature => $feature,                                       
        _entropy => $entropy,
        _class_probabilities => $class_probabilities,
        _branch_features_and_values => $branch_features_and_values,
        _linked_to => [],                                          
    }, $class;                                                     
}

# this returns the feature test at the current node
sub get_feature {                                  
    my $self = shift;                              
    return $self->{ _feature };                    
}

sub set_feature {
    my $self = shift;
    my $feature = shift;
    $self->{_feature} = $feature;
}

sub get_entropy {                                  
    my $self = shift;                              
    return $self->{ _entropy };                    
}

sub get_class_probabilities {                                  
    my $self = shift;                              
    return $self->{ _class_probabilities };                    
}

sub get_branch_features_and_values {    
    my $self = shift;                   
    return $self->{ _branch_features_and_values };     
}

sub add_to_branch_features_and_values {
    my $self = shift;                   
    my $feature_and_value = shift;
    push @{$self->{ _branch_features_and_values }}, $feature_and_value;
}

sub get_children {       
    my $self = shift;                   
    return $self->{_linked_to};
}

sub add_child_link {         
    my ($self, $new_node, ) = @_;                            
    push @{$self->{_linked_to}}, $new_node;                  
}

sub delete_all_links {                  
    my $self = shift;                   
    $self->{_linked_to} = undef;        
}

sub display_decision_tree {
    my $self = shift;
    my $offset = shift;
    if (@{$self->get_children()}) {
        my $feature_at_node = $self->get_feature() || " ";
        my $entropy_at_node = $self->get_entropy();
        my @class_probabilities = @{$self->get_class_probabilities()};
        print "NODE:  $offset  feature: $feature_at_node   entropy: $entropy_at_node  class probs: @class_probabilities\n";
        $offset = $offset . "   ";
        foreach my $child (@{$self->get_children()}) {
            $child->display_decision_tree($offset);
        }
    } else {
        my $entropy_at_node = $self->get_entropy();
        my @class_probabilities = @{$self->get_class_probabilities()};
        print "NODE:  $offset  entropy: $entropy_at_node  class probs: @class_probabilities\n";
    }
}

1;

=pod
=head1 NAME

Algorithm::DecisionTree - A pure-Perl implementation for
constructing a decision tree from multidimensional training
data and for using the decision tree thus induced for
classifying data.

=head1 SYNOPSIS

  # FOR CONSTRUCTING A DECISION TREE AND FOR CLASSIFYING A SAMPLE:

      my $training_datafile = "training.dat";
      my $dt = Algorithm::DecisionTree->new( 
                               training_datafile => $training_datafile,
      );
      $dt->get_training_data();
      $dt->show_training_data();
      my $root_node = $dt->construct_decision_tree_classifier();
      $root_node->display_decision_tree("   ");
      my @test_sample = qw /exercising=>never 
                            smoking=>heavy 
                            fatIntake=>heavy 
                            videoAddiction=>heavy /;
      $dt->classify($root_node, @test_sample);

  # For the above calls to work, the format in which the training data is made
  # available to the decision-tree constructor new() must meet certain 
  # assumptions.  (See the training.dat file in the examples directory.) The
  # training datafile must declare the class names, the feature names and 
  # the names of the different possible values for the features.  The rest of
  # the training datafile is expected to contain the training samples in the 
  # form of a multi-column table.


  # FOR GENERATING TRAINING DATA:

      use Algorithm::DecisionTree;
      my $parameter_file = "param.txt";
      my $output_data_file = "training.dat";
      my $training_data_gen = Algorithm::DecisionTree->training_data_generator( 
                                  output_datafile => $output_data_file,
                                  parameter_file    => $parameter_file,
                                  number_of_training_samples => 35,
      );
      $training_data_gen->read_parameter_file();
      $training_data_gen->gen_training_data();
      $training_data_gen->write_training_data_to_file(); 

  # For the above calls to work, the parameter file must obey certain 
  # assumptions.  (See the param.txt file in the examples directory.) The
  # parameter file must declare the class names, the class priors, the 
  # feature names and the different possible values for the features.
  # The parameter file is also expected to present information on how
  # you want the data vectors to be biased for the different classes.


  # FOR GENERATING TEST DATA:

      use Algorithm::DecisionTree;
      my $parameter_file = "param.txt";
      my $output_test_datafile = "testdata.dat";
      my $output_class_label_file = "test_data_class_labels.dat";
      my $test_data_gen = Algorithm::DecisionTree->test_data_generator(
                   output_test_datafile    => $output_test_datafile,
                   output_class_label_file => $output_class_label_file,
                   parameter_file          => $parameter_file,
                   write_to_file           => 1,
                   number_of_test_samples  => 10,
                   debug1                  => 1,
      );
      $test_data_gen->read_parameter_file();
      $test_data_gen->gen_test_data();
      $test_data_gen->write_test_data_to_file();

  # The test data is deposited without the class labels in the file named for 
  # the parameter output_test_datafile.  The class labels are deposited 
  # in a separate file named for the parameter output_class_label_file.  The
  # class names, the feature names, the feature values, and the probabilistic
  # bias used for the test data are according to the information placed in
  # the parameter file.

=head1 CHANGES

In addition to the removal of a couple of serious bugs,
version 1.2 incorporates a number of enhancements: (1)
Version 1.2 includes checks on the names of the features and
values used in test data --- this is the data you want to
classify with the decision tree classifier constructed by
this module.  (2) Version 1.2 includes a separate
constructor for generating test data.  To make it easier to
generate test data whose probabilistic parameters may not be
identical to that used for the training data, I have used
separate routines for generating the test data.  (3) Version
1.2 also includes in its examples directory a script that
classifies the test data in a file and outputs the class
labels into another file.  This is for folks who do not wish
to write their own scripts using this module. (4) Version
1.2 also includes addition to the documentation regarding
the issue of numeric values for features.

With Version 1.1, a call to classify() now returns a hash of
the class labels and their associated probabilities.
(Previously, these results were just printed out in the
terminal window.) Now you should be able to write your own
script that reads in the test data from a file and outputs
the classification results for each data vector.  This
version also includes some additional documentation and a
general cleanup of the code.

=head1 DESCRIPTION

B<Algorithm::DecisionTree> is a I<perl5> module for
constructing a decision tree from a training datafile
containing multidimensional data.  In one form or another,
decision trees have been around for about fifty years. But
their popularity during the last decade is owing to the
entropy-based method proposed by Ross Quinlan for their
construction.  Fundamental to Quinlan's approach is the
notion that a decision node in a tree should be split only
if the entropy at the ensuing child nodes will be less than
the entropy at the node in question.  The implementation
presented here is based on the same idea.

For those not familiar with decision tree ideas, the
traditional way to classify multidimensional data is to
start with a feature space whose dimensionality is the same
as that of the data.  Each feature in this space would
correspond to the attribute that each dimension of the data
measures.  You then use the training data to carve up the
feature space into different regions, each corresponding to
a different class.  Subsequently, when you are trying to
classify a new data sample, you locate it in the feature
space and find the class label of the region to which it
belongs.  One can also give the data point the same class
label as that of the nearest training sample.  (This is
referred to as the nearest neighbor classification.)

A decision tree classifier works differently.  When you
construct a decision tree, you select for the root node a
feature test that can be expected to maximally
disambiguate the class labels that could be associated with
the data you are trying to classify.  You then attach to the
root node a set of child nodes, one for each value of the
feature you chose at the root node. Now at each child node
you pose the same question that you posed when you found the
best feature to use at the root node: What feature at the
child node in question would maximally disambiguate the
class labels to be associated with a given data vector
assuming that the data vector passed the root node on the
branch that corresponds to the child node in question.  The
feature that is best at each node is the one that causes the
maximal reduction in class entropy at that node.

As the reader would expect, the two key steps in any
approach to decision-tree based classification are the
construction of the decision tree itself from a file
containing the training data, and then using the decision
tree thus obtained for classifying the data.

In addition to the above two key steps, the implementation
presented here also allows you to generate your own training
data. Generating your own training data, using it for
constructing a decision-tree classifier and subsequently
testing the classifier on a test set of data is a good way
to develop greater proficiency with decision trees.

What is cool about decision tree classification is that it
gives you soft classification, meaning it may associate more
than one class label with a given data vector.  When this
happens, it may mean that your classes are indeed
overlapping in the underlying feature space.  It could also
mean that you simply have not supplied sufficient training
data to the decision tree classifier.

For a tutorial introduction to how a decision tree is
constructed and used, please visit

L<http://cobweb.ecn.purdue.edu/~kak/DecisionTreeClassifiers.pdf>

=head1 WHAT PRACTICAL PROBLEM IS SOLVED BY THIS MODULE

Consider the following scenario: Let's say you are running a
small investment company that employs a team of
stockbrokers who make buy/sell decisions for the customers
of your company.  Assume that your company has asked the
traders to make each investment decision on the basis of the
following four criteria:

  price_to_earnings_ratio   (P_to_E)

  price_to_sales_ratio      (P_to_S)

  return_on_equity          (R_on_E)

  market_share              (MS)

Since you are the boss, you keep track of the buy/sell
decisions made by the individual traders.  But one
unfortunate day, all of your traders decide to quit because
you did not pay them enough.  So what do you do?  If you had
a module like the one here, you could still run your company
and do so in such a way that, on the average, would do
better than any of the individual traders who worked for
your company.  This is what you do: You pool together the
individual trader buy/sell decisions you have accumulated
during the last one year.  This pooled information is likely
to look like:


  example      buy/sell     P_to_E     P_to_S     R_on_E      MS
  ============================================================+=

  example_1     buy          high       low        medium    low
  example_2     buy          medium     medium     low       low
  example_3     sell         low        medium     low       high
  ....
  ....

This data would constitute your training file. You could feed this
file into the module by calling: 

    my $dt = Algorithm::DecisionTree->new( 
                                          training_datafile => $training_datafile,
                                         );
    $dt->get_training_data(); 

and then construct a decision tree by calling:

    my $root_node = $dt->construct_decision_tree_classifier();

Now you and your company (with practically no employees) are
ready to service the customers again. Suppose your computer
needs to make a buy/sell decision about an investment
prospect that is best described by:

    price_to_earnings_ratio   =>  low
    price_to_sales_ratio      =>  very_low
    return_on_equity          =>  none
    market_share              =>  medium    

All that your computer would need to do would be to
construct a data vector like

   my @data =   qw / P_to_E=>low
                     P_to_S=>very_low
                     R_on_E=>none
                     MS=>medium /;

and call the decision tree classifier you just constructed by

    $dt->classify($root_node, @data); 

The answer returned will be 'buy' and 'sell', along with the
associated probabilities.  So if the probability of 'buy' is
considerably greater than the probability of 'sell', that's
what you should instruct your computer to do.

The chances are that, on the average, this approach would
beat the performance of any of your individual traders who
worked for you previously since the buy/sell decisions made
by the computer would be based on the collective wisdom of
all your previous traders. 

B<DISCLAIMER: There is obviously a lot more to good
investing than what is captured by the silly little example
here. However, it does the convey the sense in which the
current module could be used.>

=head1 WHAT HAPPENS WHEN THE FEATURE VALUES ARE NUMERIC

The current module will treat a numeric value for a feature
as just a string.  In that sense, there is no difference
between a string value for a feature and a numeric value.
This would obviously make the module unsuitable for
applications in which a feature may take on a numeric value
from a very large set of such values and you want feature
values to be compared using numeric comparison predicates as
opposed to string comparison predicates.  (Consider, for
example, using color as an object feature in a computer
vision application.)  The decision trees for applications in
which the feature values are primarily numerical in nature
are constructed differently, as explained in the tutorial at
L<http://cobweb.ecn.purdue.edu/~kak/DecisionTreeClassifiers.pdf>


=head1 METHODS

The module provides the following methods for decision-tree
induction from training data in a diskfile, for data
classification with the decision tree, and for generating
your own training data:

=over

=item B<new():>

    my $dt = Algorithm::DecisionTree->new( 
                                          training_datafile => $training_datafile,
                                         );

A call to new() constructs a new instance of the
Algorithm::DecisionTree class.  For this call to make sense,
the training data in the training datafile must be according
to a certain format that is shown below.  (Also see the file
training.dat in the examples directory.)

=item B<get_training_data():>

After you have constructed a new instance of the Algorithm::DecisionTree
class, you must now read in the training data that is contained in the
file named above.  This you do by:

    $dt->get_training_data(); 

IMPORTANT: The training data file must in a format that
makes sense to the decision tree constructor.  The
information in this file should look like

    Class names: malignant benign

    Feature names and their values:
        videoAddiction => none low medium heavy
        exercising => never occasionally regularly
        smoking => heavy medium light never
        fatIntake => low medium heavy


    Training Data:

    sample     class      videoAddiction   exercising    smoking   fatIntake
    ==========================================================================

    sample_0   benign     medium           occasionally  heavy     low
    sample_1   malignant  none             occasionally  heavy     medium
    sample_2   benign     low              occasionally  light     heavy
    sample_3   malignant  medium           occasionally  heavy     heavy
    ....
    ....


IMPORTANT: Note that the class names, the number of classes,
the feature names, and the possible values for the features
can be anything that your data requires them to be.  The
training data file that is generated by the data generation
part of the module will be in the format shown above.  More
on that later.


=item B<show_training_data():>

If you wish to see the training data that was just digested by the module,
call 

    $dt->show_training_data(); 


=item B<construct_decision_tree_classifier():>

After the training data is digested, it is time to construct 
a decision tree classifier.  This you do by

    my $root_node = $dt->construct_decision_tree_classifier();

This call returns an instance of type Node.  The Node class is
defined within the main package file, at its end.  So, don't 
forget, that $root_node in the above example call will be
instantiated to an instance of type Node.

=item B<$root_nodeC<< -> >>display_decision_tree(" "):>

    $root_node->display_decision_tree("   ");

This will display the decision tree in your terminal window
by using a recursively determined offset for each node as
the display routine descends down the tree.

I have intentionally left the syntax fragment $root_node in
the above call to remind the reader that
display_decision_tree() is NOT called on the instance of the
DecisionTree we constructed earlier, but on the Node
instance returned by the call to
construct_decision_tree_classifier().

=item B<classify($root_node, @test_sample):>

    my @test_sample = qw /exercising=>never 
                          smoking=>heavy 
                          fatIntake=>heavy 
                          videoAddiction=>heavy /;

    my $classification = $dt->classify($root_node, @test_sample);

where, again, $root_node is an instance of type Node returned
by the call to construct_decision_tree_classifier().  The variable
$classification holds a reference to a hash whose keys are the
class labels and whose values the associated probabilities.

=item B<training_data_generator():>

The training data generator is created by using its own constructor:

    my $parameter_file = "param2.txt";
    my $output_data_file = "training.dat";
    my $training_data_gen = Algorithm::DecisionTree->training_data_generator( 
                              output_datafile => $output_data_file,
                              parameter_file    => $parameter_file,
                              number_of_training_samples => 35,
    );

=item B<$training_data_genC<< -> >>read_parameter_file():>

After you have constructed an instance of the training data
generator, you need to ask it to read the parameter file:

    $training_data_gen->read_parameter_file();

The parameter file is expected to be in the following format:

    # comment lines begin with the hash mark

    class names:  malignant benign
    class priors: 0.4 0.6

    feature: smoking
    values: heavy medium light never

    feature: exercising
    values: never occasionally regularly

    feature: fatIntake
    values: low medium heavy

    feature: videoAddiction
    values:  none low medium heavy


    bias:  class: malignant 

          smoking:    heavy=0.7
          exercising: never=0.7 
          fatIntake:  heavy=0.5
          videoAddiction: 

    bias:  class: benign

          smoking:     heavy=0.1
          exercising:  regularly=0.7
          fatIntake:   low=0.6
          videoAddiction: 


See the parameter file param.txt in the example directory.
Initially, it might be best to modify that file to suit your
needs.

IMPORTANT: You can use any names for the classes, can have
any number of classes, can use any names for the features
and their values.  

Also note the the important role played by the biasing
information.  Without the biasing, your training data will
be uniformly distributed with respect to all of the feature
values and you will only get ambiguous classifications from
the resulting decision tree classifier.  The biasing allows
you to express a higher or lower probability that a
particular feature value should have for a given class.  The
probability weight that is unused for each feature is
distributed uniformly amongst the remaining feature values.
I did experiment with the idea of assigning probability
weights to multiple (or even all) of the values for a given
feature --- it does not add to the educational value you
derive from the resulting training data.

NOTE: if you do NOT express a bias for a feature (as is the
case with the feature 'videoAddiction' above), equal weight
is given to all its values.

=item B<$training_data_genC<< -> >>gen_training_data():>

This call generators the training data from your parameter
file:

    $training_data_gen->gen_training_data();

=item B<$training_data_genC<< -> >>write_training_data_to_file():>

To write out the training data to a disk file:

    $training_data_gen->write_training_data_to_file();

This call will also display the training data in your 
terminal window if the $debug1 is on.

=item B<test_data_generator():>

The test data is generated by using its own constructor:

    my $parameter_file = "param.txt";
    my $output_test_datafile = "testdata1.dat";
    my $output_class_label_file = "test_data_class_labels.dat";

    my $test_data_gen = Algorithm::DecisionTree->test_data_generator(
                       output_test_datafile    => $output_test_datafile,
                       output_class_label_file => $output_class_label_file,
                       parameter_file          => $parameter_file,
                       write_to_file           => 1,
                       number_of_test_samples  => 10,
    );

=item B<$test_data_genC<< -> >>read_parameter_file():>

After you have constructed an instance of the test data
generator, you need to ask it to read the parameter file.

    $test_data_gen->read_parameter_file();

This parameter file named in the call to the test-data
generator constructor must possess the same structure as for
generating the training data.  In most cases, you would want
to use the same paramter file both for generating the training
data and the test data.

=item B<$test_data_genC<< -> >>gen_test_data():>

This call generates the test data from your parameter file:

    $training_data_gen->gen_training_data();

=item B<$test_data_genC<< -> >>write_test_data_to_file():>

To write out the test data to a disk file:

    $test_data_gen->write_test_data_to_file();

This call will also display the test data in your terminal
window if the $debug1 is on.


=back

=head1 HOW THE CLASSIFICATION RESULTS ARE DISPLAYED

It depends on whether you apply the classifier at once to
all the data samples in a file, or whether you feed one data
sample at a time into the classifier.

For large test datasets, you would obviously want to process
an entire file of test data at a time.  The best way to do
this is to follow my script

    classify_test_data_in_a_file.pl

in the examples directly.  This script requires three
command-line arguments, the first argument names the
training datafile, the second the test datafile, and the
third in which the classification results will be deposited.

You can also invoke the classifier on one data sample at a
time.  A call such as

    my @test_sample = qw /exercising=>never 
                          smoking=>heavy 
                          fatIntake=>heavy 
                          videoAddiction=>heavy /;

    my $classification = $dt->classify($root_node, @test_sample);
    print "The classification:\n";
    foreach my $class ($dt->get_class_names()) {
        print "    $class with probability $classification->{$class}\n"; 
    }    

will print out the classification results in the following form:

    The classification:
        malignant with probability 0.744186046511628
        benign with probability 0.255813953488372

Note again the soft classification returned.  That is, if
the probability distributions for the different classes
overlap in the underlying feature space, the classifier will
return all of the applicable class labels for a data vector
along with the corresponding class probabilities.  Another
reason for why the decision tree classifier may associate
significant probabilities with multiple class labels is that
you used inadequate number of training samples to induce the
decision tree.  B<The good thing is that the classifier does
not lie to you> (unlike, say, a hard classification rule
that would corresponding to a partitioning of the underlying
feature space).  The decision tree classifier give you the
best classification that can be made given the training data
you fed into it.

=head1 EXAMPLES

See the examples directory in the distribution for how to
generate the training data, how to induce a decision tree,
and how to then classify new data using the decision tree.

To become more familiar with the module, run the script

    training_data_generator.pl

to generate a training datafile according to the information
placed in the file param.txt and then run the script 

    construct_dt_and_classify_one_sample.pl

to classify a new data sample that is in the script.  Next
generate a test dataset by calling

    generate_test_data.pl*

This will deposit the test data in a file.  You can invoke the
classifier on this file by an invocation like

    classify_test_data_in_a_file.pl   training.dat   testdata2.dat   out.txt

=head1 EXPORT

None by design.

=head1 BUGS

Please notify the author if you encounter any bugs.  When
sending email, please place the string 'DecisionTree' in the
subject line.

=head1 INSTALLATION

The usual

    perl Makefile.PL
    make
    make test
    make install

if you have root access.  If not, 

    perl Makefile.PL prefix=/some/other/directory/
    make
    make test
    make install

=head1 AUTHOR

Avinash Kak, kak@purdue.edu

If you send email, please place the string "DecisionTree" in your
subject line to get past my spam filter.

=head1 COPYRIGHT

This library is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

 Copyright 2010 Avinash Kak

=cut

