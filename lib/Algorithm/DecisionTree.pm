package Algorithm::DecisionTree;

#--------------------------------------------------------------------------------------
# Copyright (c) 2014 Avinash Kak. All rights reserved.
# This program is free software.  You may modify and/or
# distribute it under the same terms as Perl itself.
# This copyright notice must remain attached to the file.
#
# Algorithm::DecisionTree is a Perl module for constructing
# a decision tree from training examples of multidimensional
# data and then using the tree thus constructed to classify
# new data.
#---------- ---------------------------------------------------------------------------

use 5.18.0;
use strict;
use warnings;
use Carp;

our $VERSION = '2.22';

############################################   Constructor  ##############################################

# Constructor:
sub new { 
    my ($class, %args, $evalmode_or_not);
    if (@_ % 2 != 0) {
        ($class, %args) = @_;
    } else {
        $class = shift;
        $evalmode_or_not = shift;
        die "Only one string arg allowed in evalmode" if @_;
    }
    unless ($evalmode_or_not) {
        my @params = keys %args;
        croak "\nYou have used a wrong name for a keyword argument --- perhaps a misspelling\n" 
                           if check_for_illegal_params2(@params) == 0;
    }
    bless {
        _training_datafile                   =>    $args{training_datafile}, 
        _entropy_threshold                   =>    $args{entropy_threshold} || 0.01,
        _max_depth_desired                   =>    $args{max_depth_desired} || undef,
        _debug1                              =>    $args{debug1} || 0,
        _debug2                              =>    $args{debug2} || 0,
        _debug3                              =>    $args{debug3} || 0,
        _csv_class_column_index              =>    $args{csv_class_column_index} || undef,
        _csv_columns_for_features            =>    $args{csv_columns_for_features} || undef,
        _symbolic_to_numeric_cardinality_threshold
                                             =>    $args{symbolic_to_numeric_cardinality_threshold} || 10,
        _number_of_histogram_bins            =>    $args{number_of_histogram_bins} || undef,
        _root_node                           =>    undef,
        _probability_cache                   =>    {},
        _entropy_cache                       =>    {},
        _training_data_hash                  =>    {},
        _features_and_values_hash            =>    {},
        _samples_class_label_hash            =>    {},
        _class_names                         =>    [],
        _class_priors                        =>    [],
        _feature_names                       =>    [],
        _numeric_features_valuerange_hash    =>    {},
        _sampling_points_for_numeric_feature_hash      =>      {},
        _feature_values_how_many_uniques_hash          =>      {},
        _prob_distribution_numeric_features_hash       =>      {},
        _histogram_delta_hash                          =>      {},
        _num_of_histogram_bins_hash                    =>      {},
    }, $class;

}

####################################  Classify with Decision Tree  #######################################

##  Classifies one test sample at a time using the decision tree constructed from
##  your training file.  The data record for the test sample must be supplied as
##  shown in the scripts in the `examples' subdirectory.  See the scripts
##  construct_dt_and_classify_one_sample_caseX.pl in that subdirectory.
sub classify {
    my $self = shift;
    my $root_node = shift;
    my $feature_and_values = shift;
    my $numregex =  '[+-]?\ *(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?';
    my @features_and_values = @$feature_and_values;
    @features_and_values = @{deep_copy_array(\@features_and_values)};
    die "Error in the names you have used for features and/or values" 
                        unless $self->check_names_used(\@features_and_values);
    my @new_features_and_values = ();
    my $pattern = '(\S+)\s*=\s*(\S+)';
    foreach my $feature_and_value (@features_and_values) {
        $feature_and_value =~ /$pattern/;
        my ($feature, $value) = ($1, $2);
        my $newvalue = $value;
        my @unique_values_for_feature = @{$self->{_features_and_unique_values_hash}->{$feature}};
        my $not_all_values_float = 0;
        # map {$not_all_values_float = 1 if $_ !~ /^\d*\.\d+$/} @unique_values_for_feature;
        map {$not_all_values_float = 1 if $_ !~ /^$numregex$/} @unique_values_for_feature;
        if (! contained_in($feature, keys %{$self->{_prob_distribution_numeric_features_hash}}) &&
                                                                       $not_all_values_float == 0) {
            $newvalue = closest_sampling_point($value, \@unique_values_for_feature);
        }
        push @new_features_and_values, "$feature" . '=' . "$newvalue";
    }
    @features_and_values = @new_features_and_values;
    print "\nCL1 New feature and values: @features_and_values\n" if $self->{_debug3};
    my %answer = ();
    foreach my $class_name (@{$self->{_class_names}}) {
        $answer{$class_name} = undef;
    }
    $answer{'solution_path'} = [];
    my %classification = %{$self->recursive_descent_for_classification($root_node, 
                                                                    \@features_and_values,\%answer)};
    @{$answer{'solution_path'}} = reverse @{$answer{'solution_path'}};
    if ($self->{_debug3}) {
        print "\nCL2 The classification:\n";
        foreach my $class_name (@{$self->{_class_names}}) {
            print "    $class_name  with probability $classification{$class_name}\n";
        }
    }
    my %classification_for_display = ();
    foreach my $item (keys %classification) {
        if ($item ne 'solution_path') {
            $classification_for_display{$item} = sprintf("%0.3f", $classification{$item});
        } else {
            my @outlist = ();
            foreach my $x (@{$classification{$item}}) {
                push @outlist, "NODE$x";
            }
            $classification_for_display{$item} =  \@outlist;
        }
    }
    return \%classification_for_display;
}

sub recursive_descent_for_classification {
    my $self = shift;
    my $node = shift;
    my $features_and_values = shift;
    my $answer = shift;
    my @features_and_values = @$features_and_values;
    my %answer = %$answer;
    my @children = @{$node->get_children()};
    if (@children == 0) {
        my @leaf_node_class_probabilities = @{$node->get_class_probabilities()};
        foreach my $i (0..@{$self->{_class_names}}-1) {
            $answer{$self->{_class_names}->[$i]} = $leaf_node_class_probabilities[$i];
        }
        push @{$answer{'solution_path'}}, $node->get_serial_num();
        return \%answer;
    }
    my $feature_tested_at_node = $node->get_feature();
    print "\nCLRD1 Feature tested at node for classification: $feature_tested_at_node\n" 
        if $self->{_debug3};
    my $value_for_feature;
    my $path_found;
    my $pattern = '(\S+)\s*=\s*(\S+)';
    foreach my $feature_and_value (@features_and_values) {
        $feature_and_value =~ /$pattern/;
        $value_for_feature = $2 if $feature_tested_at_node eq $1;
    }
    if ($value_for_feature) {
        if (contained_in($feature_tested_at_node, keys %{$self->{_prob_distribution_numeric_features_hash}})) {
            print( "\nCLRD2 In the truly numeric section") if $self->{_debug3};
            my $pattern1 = '(.+)<(.+)';
            my $pattern2 = '(.+)>(.+)';
            foreach my $child (@children) {
                my @branch_features_and_values = @{$child->get_branch_features_and_values_or_thresholds()};
                my $last_feature_and_value_on_branch = $branch_features_and_values[-1]; 
                if ($last_feature_and_value_on_branch =~ /$pattern1/) {
                    my ($feature, $threshold) = ($1,$2); 
                    if ($value_for_feature <= $threshold) {
                        $path_found = 1;
                        %answer = %{$self->recursive_descent_for_classification($child,
                                                                             $features_and_values,\%answer)};
                        push @{$answer{'solution_path'}}, $node->get_serial_num();
                        last;
                    }
                }
                if ($last_feature_and_value_on_branch =~ /$pattern2/) {
                    my ($feature, $threshold) = ($1,$2); 
                    if ($value_for_feature > $threshold) {
                        $path_found = 1;
                        %answer = %{$self->recursive_descent_for_classification($child,
                                                                            $features_and_values,\%answer)};
                        push @{$answer{'solution_path'}}, $node->get_serial_num();
                        last;
                    }
                }
            }
            return \%answer if $path_found;
        } else {
            my $feature_value_combo = "$feature_tested_at_node" . '=' . "$value_for_feature";
            print "\nCLRD3 In the symbolic section with feature_value_combo: $feature_value_combo\n" 
                if $self->{_debug3};
            foreach my $child (@children) {
                my @branch_features_and_values = @{$child->get_branch_features_and_values_or_thresholds()};
                print "\nCLRD4 branch features and values: @branch_features_and_values\n" if $self->{_debug3};
                my $last_feature_and_value_on_branch = $branch_features_and_values[-1]; 
                if ($last_feature_and_value_on_branch eq $feature_value_combo) {
                    %answer = %{$self->recursive_descent_for_classification($child,
                                                                              $features_and_values,\%answer)};
                    push @{$answer{'solution_path'}}, $node->get_serial_num();
                    $path_found = 1;
                    last;
                }
            }
            return \%answer if $path_found;
        }
    }
    if (! $path_found) {
        my @leaf_node_class_probabilities = @{$node->get_class_probabilities()};
        foreach my $i (0..@{$self->{_class_names}}-1) {
            $answer{$self->{_class_names}->[$i]} = $leaf_node_class_probabilities[$i];
        }
        push @{$answer{'solution_path'}}, $node->get_serial_num();
    }
    return \%answer;
}

##  If you want classification to be carried out by engaging a human user in a
##  question-answer session, this is the method to use for that purpose.  See, for
##  example, the script classify_by_asking_questions.pl in the `examples'
##  subdirectory for an illustration of how to do that.
sub classify_by_asking_questions {
    my $self = shift;
    my $root_node = shift;
    my %answer = ();
    foreach my $class_name (@{$self->{_class_names}}) {
        $answer{$class_name} = undef;
    }
    $answer{'solution_path'} = [];
    my %scratchpad_for_numeric_answers = ();
    foreach my $feature_name (keys %{$self->{_prob_distribution_numeric_features_hash}}) {
        $scratchpad_for_numeric_answers{$feature_name} = undef;
    }
    my %classification = %{$self->interactive_recursive_descent_for_classification($root_node,
                                                       \%answer, \%scratchpad_for_numeric_answers)};
    @{$classification{'solution_path'}} = reverse @{$classification{'solution_path'}};
    my %classification_for_display = ();
    foreach my $item (keys %classification) {
        if ($item ne 'solution_path') {
            $classification_for_display{$item} = sprintf("%0.3f", $classification{$item});
        } else {
            my @outlist = ();
            foreach my $x (@{$classification{$item}}) {
                push @outlist, "NODE$x";
            }
            $classification_for_display{$item} =  \@outlist;
        }
    }
    return \%classification_for_display;
}

sub interactive_recursive_descent_for_classification {
    my $self = shift;
    my $node = shift;
    my $answer = shift;
    my $scratchpad_for_numerics = shift;
    my %answer = %$answer;
    my %scratchpad_for_numerics = %$scratchpad_for_numerics;
    my $pattern1 = '(.+)<(.+)';
    my $pattern2 = '(.+)>(.+)';
    my $user_value_for_feature;
    my @children = @{$node->get_children()};
    if (@children == 0) {
        my @leaf_node_class_probabilities = @{$node->get_class_probabilities()};
        foreach my $i (0..@{$self->{_class_names}}-1) {
            $answer{$self->{_class_names}->[$i]} = $leaf_node_class_probabilities[$i];
        }
        push @{$answer{'solution_path'}}, $node->get_serial_num();
        return \%answer;
    }
    my @list_of_branch_attributes_to_children = ();
    foreach my $child (@children) {   
        my @branch_features_and_values = @{$child->get_branch_features_and_values_or_thresholds()};
        my $feature_and_value_on_branch = $branch_features_and_values[-1];
        push @list_of_branch_attributes_to_children, $feature_and_value_on_branch;
    }
    my $feature_tested_at_node = $node->get_feature();
    my $feature_value_combo;
    my $path_found = 0;
    if (contained_in($feature_tested_at_node, keys %{$self->{_prob_distribution_numeric_features_hash}})) {
        if ($scratchpad_for_numerics{$feature_tested_at_node}) {
            $user_value_for_feature = $scratchpad_for_numerics{$feature_tested_at_node};
        } else {
            my @valuerange =  @{$self->{_numeric_features_valuerange_hash}->{$feature_tested_at_node}};
            while (1) { 
                print "\nWhat is the value for the feature $feature_tested_at_node ?\n";
                print "\nEnter a value in the range (@valuerange): ";
                $user_value_for_feature = <STDIN>;
                chomp $user_value_for_feature;
                $user_value_for_feature =~ s/^\s*(\S+)\s*$/$1/;
                my $answer_found = 0;
                if ($user_value_for_feature >= $valuerange[0] && $user_value_for_feature <= $valuerange[1]) {
                    $answer_found = 1;
                    last;
                }
                last if $answer_found;
                print("You entered illegal value. Let's try again")
            }
            $scratchpad_for_numerics{$feature_tested_at_node} = $user_value_for_feature;
        }
        foreach my $i (0..@list_of_branch_attributes_to_children-1) {
            my $branch_attribute = $list_of_branch_attributes_to_children[$i];
            if ($branch_attribute =~ /$pattern1/) {
                my ($feature,$threshold) = ($1,$2);
                if ($user_value_for_feature <= $threshold) {
                    %answer = %{$self->interactive_recursive_descent_for_classification($children[$i],
                                                                     \%answer, \%scratchpad_for_numerics)};
                    $path_found = 1;
                    push @{$answer{'solution_path'}}, $node->get_serial_num();
                    last;
                }
            }
            if ($branch_attribute =~ /$pattern2/) {
                my ($feature,$threshold) = ($1,$2);
                if ($user_value_for_feature > $threshold) {
                    %answer = %{$self->interactive_recursive_descent_for_classification($children[$i],
                                                                     \%answer, \%scratchpad_for_numerics)};
                    $path_found = 1;
                    push @{$answer{'solution_path'}}, $node->get_serial_num();
                    last;
                }
            }
        }
        return \%answer if $path_found;
    } else {
        my @possible_values_for_feature = @{$self->{_features_and_unique_values_hash}->{$feature_tested_at_node}};
        while (1) {
            print "\nWhat is the value for the feature $feature_tested_at_node ?\n";
            print "\nEnter a value from the list (@possible_values_for_feature): ";
            $user_value_for_feature = <STDIN>;
            chomp $user_value_for_feature;
            $user_value_for_feature =~ s/^\s*(\S+)\s*$/$1/;
            my $answer_found = 0;
            if (contained_in($user_value_for_feature, @possible_values_for_feature)) {
                $answer_found = 1;
                last;
            }
            last if $answer_found;
            print("You entered illegal value. Let's try again");
        }
        $feature_value_combo = "$feature_tested_at_node=$user_value_for_feature";
        foreach my $i (0..@list_of_branch_attributes_to_children-1) {
            my $branch_attribute = $list_of_branch_attributes_to_children[$i];
            if ($branch_attribute eq $feature_value_combo) {
                %answer = %{$self->interactive_recursive_descent_for_classification($children[$i],
                                                                     \%answer, \%scratchpad_for_numerics)};
                $path_found = 1;
                push @{$answer{'solution_path'}}, $node->get_serial_num();
                last;
            }
        }
        return \%answer if $path_found;
    }
    if (! $path_found) {
        my @leaf_node_class_probabilities = @{$node->get_class_probabilities()};
        foreach my $i (0..@{$self->{_class_names}}-1) {
            $answer{$self->{_class_names}->[$i]} = $leaf_node_class_probabilities[$i];
        }
        push @{$answer{'solution_path'}}, $node->get_serial_num();
    }
    return \%answer;
}

######################################    Decision Tree Construction  ####################################

##  At the root node, we find the best feature that yields the greatest reduction in
##  class entropy from the entropy based on just the class priors. The logic for
##  finding this feature is different for symbolic features and for numeric features.
##  That logic is built into the method shown later for best feature calculations.
sub construct_decision_tree_classifier {
    my $self = shift;
    if ($self->{_debug3}) {        
        $self->determine_data_condition(); 
        print "\nStarting construction of the decision tree:\n";
    }
    my @class_probabilities = map {$self->prior_probability_for_class($_)} @{$self->{_class_names}};
    if ($self->{_debug3}) { 
        print "\nPrior class probabilities: @class_probabilities\n";
        print "\nClass names: @{$self->{_class_names}}\n";
    }
    my $entropy = $self->class_entropy_on_priors();
    print "\nClass entropy on priors: $entropy\n" if $self->{_debug3};
    my $root_node = DTNode->new(undef, $entropy, \@class_probabilities, [], 'root');
    DTNode->set_class_names(\@{$self->{_class_names}});
    $self->{_root_node} = $root_node;
    $self->recursive_descent($root_node);
    return $root_node;
}

##  After the root node of the decision tree is calculated by the previous methods,
##  we invoke this method recursively to create the rest of the tree.  At each node,
##  we find the feature that achieves the largest entropy reduction with regard to
##  the partitioning of the training data samples that correspond to that node.
sub recursive_descent {
    my $self = shift;
    my $node = shift;
    print "\n==================== ENTERING RECURSIVE DESCENT ==========================\n"
        if $self->{_debug3};
    my $node_serial_number = $node->get_serial_num();
    my @features_and_values_or_thresholds_on_branch = @{$node->get_branch_features_and_values_or_thresholds()};
    my $existing_node_entropy = $node->get_node_entropy();
    if ($self->{_debug3}) { 
        print "\nRD1 NODE SERIAL NUMBER: $node_serial_number\n";
        print "\nRD2 Existing Node Entropy: $existing_node_entropy\n";
        print "\nRD3 features_and_values_or_thresholds_on_branch: @features_and_values_or_thresholds_on_branch\n";
        my @class_probs = @{$node->get_class_probabilities()};
        print "\nRD4 Class probabilities: @class_probs\n";
    }
    if ($existing_node_entropy < $self->{_entropy_threshold}) { 
        print "\nRD5 returning because existing node entropy is below threshold\n" if $self->{_debug3};
        return;
    }
    my @copy_of_path_attributes = @{deep_copy_array(\@features_and_values_or_thresholds_on_branch)};
    my ($best_feature, $best_feature_entropy, $best_feature_val_entropies, $decision_val) =
                    $self->best_feature_calculator(\@copy_of_path_attributes, $existing_node_entropy);
    $node->set_feature($best_feature);
    $node->display_node() if $self->{_debug3};
    if (defined($self->{_max_depth_desired}) && 
               (@features_and_values_or_thresholds_on_branch >= $self->{_max_depth_desired})) {
        print "\nRD6 REACHED LEAF NODE AT MAXIMUM DEPTH ALLOWED\n" if $self->{_debug3}; 
        return;
    }
    return if ! defined $best_feature;
    if ($self->{_debug3}) { 
        print "\nRD7 Existing entropy at node: $existing_node_entropy\n";
        print "\nRD8 Calculated best feature is $best_feature and its value $decision_val\n";
        print "\nRD9 Best feature entropy: $best_feature_entropy\n";
        print "\nRD10 Calculated entropies for different values of best feature: @$best_feature_val_entropies\n";
    }
    my $entropy_gain = $existing_node_entropy - $best_feature_entropy;
    print "\nRD11 Expected entropy gain at this node: $entropy_gain\n" if $self->{_debug3};
    if ($entropy_gain > $self->{_entropy_threshold}) {
        if (exists $self->{_numeric_features_valuerange_hash}->{$best_feature} && 
              $self->{_feature_values_how_many_uniques_hash}->{$best_feature} > 
                                        $self->{_symbolic_to_numeric_cardinality_threshold}) {
            my $best_threshold = $decision_val;            # as returned by best feature calculator
            my ($best_entropy_for_less, $best_entropy_for_greater) = @$best_feature_val_entropies;
            my @extended_branch_features_and_values_or_thresholds_for_lessthan_child = 
                                        @{deep_copy_array(\@features_and_values_or_thresholds_on_branch)};
            my @extended_branch_features_and_values_or_thresholds_for_greaterthan_child  = 
                                        @{deep_copy_array(\@features_and_values_or_thresholds_on_branch)}; 
            my $feature_threshold_combo_for_less_than = "$best_feature" . '<' . "$best_threshold";
            my $feature_threshold_combo_for_greater_than = "$best_feature" . '>' . "$best_threshold";
            push @extended_branch_features_and_values_or_thresholds_for_lessthan_child, 
                                                                  $feature_threshold_combo_for_less_than;
            push @extended_branch_features_and_values_or_thresholds_for_greaterthan_child, 
                                                               $feature_threshold_combo_for_greater_than;
            if ($self->{_debug3}) {
                print "\nRD12 extended_branch_features_and_values_or_thresholds_for_lessthan_child: " .
                      "@extended_branch_features_and_values_or_thresholds_for_lessthan_child\n";
                print "\nRD13 extended_branch_features_and_values_or_thresholds_for_greaterthan_child: " .
                      "@extended_branch_features_and_values_or_thresholds_for_greaterthan_child\n";
            }
            my @class_probabilities_for_lessthan_child_node = 
                map {$self->probability_of_a_class_given_sequence_of_features_and_values_or_thresholds($_,
                 \@extended_branch_features_and_values_or_thresholds_for_lessthan_child)} @{$self->{_class_names}};
            my @class_probabilities_for_greaterthan_child_node = 
                map {$self->probability_of_a_class_given_sequence_of_features_and_values_or_thresholds($_,
              \@extended_branch_features_and_values_or_thresholds_for_greaterthan_child)} @{$self->{_class_names}};
            if ($self->{_debug3}) {
                print "\nRD14 class entropy for going down lessthan child: $best_entropy_for_less\n";
                print "\nRD15 class_entropy_for_going_down_greaterthan_child: $best_entropy_for_greater\n";
            }
            if ($best_entropy_for_less < $existing_node_entropy - $self->{_entropy_threshold}) {
                my $left_child_node = DTNode->new(undef, $best_entropy_for_less,
                                                         \@class_probabilities_for_lessthan_child_node,
                                   \@extended_branch_features_and_values_or_thresholds_for_lessthan_child);
                $node->add_child_link($left_child_node);
                $self->recursive_descent($left_child_node);
            }
            if ($best_entropy_for_greater < $existing_node_entropy - $self->{_entropy_threshold}) {
                my $right_child_node = DTNode->new(undef, $best_entropy_for_greater,
                                                         \@class_probabilities_for_greaterthan_child_node,
                                \@extended_branch_features_and_values_or_thresholds_for_greaterthan_child);
                $node->add_child_link($right_child_node);
                $self->recursive_descent($right_child_node);
            }
        } else {
            print "\nRD16 RECURSIVE DESCENT: In section for symbolic features for creating children"
                if $self->{_debug3};
            my @values_for_feature = @{$self->{_features_and_unique_values_hash}->{$best_feature}};
            print "\nRD17 Values for feature $best_feature are @values_for_feature\n" if $self->{_debug3};
            my @feature_value_combos = sort map {"$best_feature" . '=' . $_} @values_for_feature;
            my @class_entropies_for_children = ();
            foreach my $feature_and_value_index (0..@feature_value_combos-1) {
                print "\nRD18 Creating a child node for: $feature_value_combos[$feature_and_value_index]\n"
                    if $self->{_debug3};
                my @extended_branch_features_and_values_or_thresholds;
                if (! @features_and_values_or_thresholds_on_branch) {
                    @extended_branch_features_and_values_or_thresholds = 
                                                          ($feature_value_combos[$feature_and_value_index]);
                } else {
                    @extended_branch_features_and_values_or_thresholds = 
                        @{deep_copy_array(\@features_and_values_or_thresholds_on_branch)};
                    push @extended_branch_features_and_values_or_thresholds, 
                                           $feature_value_combos[$feature_and_value_index];
                }
                my @class_probabilities =
                   map {$self->probability_of_a_class_given_sequence_of_features_and_values_or_thresholds($_,
                               \@extended_branch_features_and_values_or_thresholds)} @{$self->{_class_names}};
                my $class_entropy_for_child = 
                      $self->class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(
                                                         \@extended_branch_features_and_values_or_thresholds);
                if ($self->{_debug3}) {
                    print "\nRD19 branch attributes: @extended_branch_features_and_values_or_thresholds\n";
                    print "\nRD20 class entropy for child: $class_entropy_for_child\n"; 
                }
                if ($existing_node_entropy - $class_entropy_for_child > $self->{_entropy_threshold}) {
                    my $child_node = DTNode->new(undef, $class_entropy_for_child,
                               \@class_probabilities, \@extended_branch_features_and_values_or_thresholds);
                    $node->add_child_link($child_node);
                    $self->recursive_descent($child_node);
                } else {
                    print "\nRD21 This child will NOT result in a node\n" if $self->{_debug3};
                }
            }
        }
    } else {
        print "\nRD22 REACHED LEAF NODE NATURALLY for: @features_and_values_or_thresholds_on_branch\n" 
            if $self->{_debug3};
        return;
    }
}

##  This is the heart of the decision tree constructor.  Its main job is to figure
##  out the best feature to use for partitioning the training data samples that
##  correspond to the current node.  The search for the best feature is carried out
##  differently for symbolic features and for numeric features.  For a symbolic
##  feature, the method estimates the entropy for each value of the feature and then
##  averages out these entropies as a measure of the discriminatory power of that
##  features.  For a numeric feature, on the other hand, it estimates the entropy
##  reduction that can be achieved if were to partition the set of training samples
##  for each possible threshold.  For a numeric feature, all possible sampling points
##  relevant to the node in question are considered as candidates for thresholds.
sub best_feature_calculator {
    my $self = shift;
    my $features_and_values_or_thresholds_on_branch = shift;
    my $existing_node_entropy = shift;
    my @features_and_values_or_thresholds_on_branch =  @$features_and_values_or_thresholds_on_branch;
    my $pattern1 = '(.+)=(.+)';
    my $pattern2 = '(.+)<(.+)';
    my $pattern3 = '(.+)>(.+)';
    my @all_symbolic_features = ();
    foreach my $feature_name (@{$self->{_feature_names}}) {
        push @all_symbolic_features, $feature_name 
            if ! exists $self->{_prob_distribution_numeric_features_hash}->{$feature_name};
    }
    my @symbolic_features_already_used = ();  
    foreach my $feature_and_value_or_threshold (@features_and_values_or_thresholds_on_branch) {
        push @symbolic_features_already_used, $1 if $feature_and_value_or_threshold =~ /$pattern1/;
    }
    my @symbolic_features_not_yet_used;
    foreach my $x (@all_symbolic_features) {
        push @symbolic_features_not_yet_used, $x unless contained_in($x, @symbolic_features_already_used);
    }
    my @true_numeric_types = ();
    my @symbolic_types = ();
    my @true_numeric_types_feature_names = ();
    my @symbolic_types_feature_names = ();
    foreach my $item (@features_and_values_or_thresholds_on_branch) {
        if ($item =~ /$pattern2/) {
            push @true_numeric_types, $item;
            push @true_numeric_types_feature_names, $1;
        } elsif ($item =~ /$pattern3/) {
            push @true_numeric_types, $item;
            push @true_numeric_types_feature_names, $1;
        } elsif ($item =~ /$pattern1/) {
            push @symbolic_types, $item;
            push @symbolic_types_feature_names, $1;
        } else {
            die "format error in the representation of feature and values or thresholds";
        }
    }
    my %seen = ();
    @true_numeric_types_feature_names = grep {$_ if !$seen{$_}++} @true_numeric_types_feature_names;
    %seen = ();
    @symbolic_types_feature_names = grep {$_ if !$seen{$_}++} @symbolic_types_feature_names;
    my @bounded_intervals_numeric_types = 
                       @{$self->find_bounded_intervals_for_numeric_features(\@true_numeric_types)};
    # Calculate the upper and the lower bounds to be used when searching for the best
    # threshold for each of the numeric features that are in play at the current node:
    my (%upperbound, %lowerbound);
    foreach my $feature (@true_numeric_types_feature_names) {
        $upperbound{$feature} = undef;
        $lowerbound{$feature} = undef;
    }
    foreach my $item (@bounded_intervals_numeric_types) {
        foreach my $feature_grouping (@$item) {
            if ($feature_grouping->[1] eq '>') {
                $lowerbound{$feature_grouping->[0]} = $feature_grouping->[2];
            } else {
                $upperbound{$feature_grouping->[0]} = $feature_grouping->[2];
            }
        }
    }
    my %entropy_values_for_different_features = ();
    my %partitioning_point_child_entropies_hash = ();
    my %partitioning_point_threshold = ();
    my %entropies_for_different_values_of_symbolic_feature = ();
    foreach my $feature (@{$self->{_feature_names}}) {
        $entropy_values_for_different_features{$feature} = [];
        $partitioning_point_child_entropies_hash{$feature} = {};
        $partitioning_point_threshold{$feature} = undef;
        $entropies_for_different_values_of_symbolic_feature{$feature} = [];
    }
    foreach my $i (0..@{$self->{_feature_names}}-1) {
        my $feature_name = $self->{_feature_names}->[$i];
        print "\n\nBFC1          FEATURE BEING CONSIDERED: $feature_name\n" if $self->{_debug3};
        if (contained_in($feature_name, @symbolic_features_already_used)) {
            next;
        } elsif (contained_in($feature_name, keys %{$self->{_numeric_features_valuerange_hash}}) &&
                 $self->{_feature_values_how_many_uniques_hash}->{$feature_name} >
                                      $self->{_symbolic_to_numeric_cardinality_threshold}) {
            my @values = @{$self->{_sampling_points_for_numeric_feature_hash}->{$feature_name}};
            print "\nBFC4 values for $feature_name are @values\n" if $self->{_debug3};      
            my @newvalues = ();
            if (contained_in($feature_name, @true_numeric_types_feature_names)) {
                if (defined($upperbound{$feature_name}) && defined($lowerbound{$feature_name}) &&
                              $lowerbound{$feature_name} >= $upperbound{$feature_name}) {
                    next;
                } elsif (defined($upperbound{$feature_name}) && defined($lowerbound{$feature_name}) &&
                                    $lowerbound{$feature_name} < $upperbound{$feature_name}) {
                    foreach my $x (@values) {
                        push @newvalues, $x if $x > $lowerbound{$feature_name} && $x <= $upperbound{$feature_name};
                    }
                } elsif (defined($upperbound{$feature_name})) {
                    foreach my $x (@values) {
                        push @newvalues, $x if $x <= $upperbound{$feature_name};
                    }
                } elsif (defined($lowerbound{$feature_name})) {
                    foreach my $x (@values) {
                        push @newvalues, $x if $x > $lowerbound{$feature_name};
                    }
                } else {
                    die "Error is bound specifications in best feature calculator";
                }
            } else {
                @newvalues = @{deep_copy_array(\@values)};
            }
            next if @newvalues == 0;
            my @partitioning_entropies = ();            
            foreach my $value (@newvalues) {
                my $feature_and_less_than_value_string =  "$feature_name" . '<' . "$value";
                my $feature_and_greater_than_value_string = "$feature_name" . '>' . "$value";
                my @for_left_child;
                my @for_right_child;
                if (@features_and_values_or_thresholds_on_branch) {
                    @for_left_child = @{deep_copy_array(\@features_and_values_or_thresholds_on_branch)};
                    push @for_left_child, $feature_and_less_than_value_string;
                    @for_right_child = @{deep_copy_array(\@features_and_values_or_thresholds_on_branch)};
                    push @for_right_child, $feature_and_greater_than_value_string;
                } else {
                    @for_left_child = ($feature_and_less_than_value_string);
                    @for_right_child = ($feature_and_greater_than_value_string);
                }
                my $entropy1 = $self->class_entropy_for_less_than_threshold_for_feature(
                                    \@features_and_values_or_thresholds_on_branch, $feature_name, $value);
                my $entropy2 = $self->class_entropy_for_greater_than_threshold_for_feature(
                                    \@features_and_values_or_thresholds_on_branch, $feature_name, $value);
                my $partitioning_entropy = $entropy1 * 
                     $self->probability_of_a_sequence_of_features_and_values_or_thresholds(\@for_left_child) +
                                           $entropy2 *
                     $self->probability_of_a_sequence_of_features_and_values_or_thresholds(\@for_right_child);

                push @partitioning_entropies, $partitioning_entropy;
                $partitioning_point_child_entropies_hash{$feature_name}{$value} = [$entropy1, $entropy2];
            }
            my ($min_entropy, $best_partition_point_index) = minimum(\@partitioning_entropies);
            if ($min_entropy < $existing_node_entropy) {
                $partitioning_point_threshold{$feature_name} = $newvalues[$best_partition_point_index];
                $entropy_values_for_different_features{$feature_name} = $min_entropy;
            }
        } else {
            print "\nBFC2:  Entering section reserved for symbolic features\n" if $self->{_debug3};
            print "\nBFC3 Feature name: $feature_name\n" if $self->{_debug3};
            my %seen;
            my @values = grep {$_ ne 'NA' && !$seen{$_}++} 
                                    @{$self->{_features_and_unique_values_hash}->{$feature_name}};
            @values = sort @values;
            print "\nBFC4 values for feature $feature_name are @values\n" if $self->{_debug3};

            my $entropy = 0;
            foreach my $value (@values) {
                my $feature_value_string = "$feature_name" . '=' . "$value";
                print "\nBFC4 feature_value_string: $feature_value_string\n" if $self->{_debug3};
                my @extended_attributes = @{deep_copy_array(\@features_and_values_or_thresholds_on_branch)};
                if (@features_and_values_or_thresholds_on_branch) {
                    push @extended_attributes, $feature_value_string;
                } else {
                    @extended_attributes = ($feature_value_string);
                }
                $entropy += 
           $self->class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds(\@extended_attributes) * 
           $self->probability_of_a_sequence_of_features_and_values_or_thresholds(\@extended_attributes);
                print "\nBFC5 Entropy calculated for symbolic feature value choice ($feature_name,$value) " .
                      "is $entropy\n" if $self->{_debug3};
                push @{$entropies_for_different_values_of_symbolic_feature{$feature_name}}, $entropy;
            }
            if ($entropy < $existing_node_entropy) {
                $entropy_values_for_different_features{$feature_name} = $entropy;
            }
        }
    }
    my $min_entropy_for_best_feature;
    my $best_feature_name;
    foreach my $feature_nom (keys %entropy_values_for_different_features) { 
        if (!defined($best_feature_name)) {
            $best_feature_name = $feature_nom;
            $min_entropy_for_best_feature = $entropy_values_for_different_features{$feature_nom};
        } else {
            if ($entropy_values_for_different_features{$feature_nom} < $min_entropy_for_best_feature) {
                $best_feature_name = $feature_nom;
                $min_entropy_for_best_feature = $entropy_values_for_different_features{$feature_nom};
            }
        }
    }
    my $threshold_for_best_feature;
    if (exists $partitioning_point_threshold{$best_feature_name}) {
        $threshold_for_best_feature = $partitioning_point_threshold{$best_feature_name};
    } else {
        $threshold_for_best_feature = undef;
    }
    my $best_feature_entropy = $min_entropy_for_best_feature;
    my @val_based_entropies_to_be_returned;
    my $decision_val_to_be_returned;
    if (exists $self->{_numeric_features_valuerange_hash}->{$best_feature_name} && 
          $self->{_feature_values_how_many_uniques_hash}->{$best_feature_name} > 
                                    $self->{_symbolic_to_numeric_cardinality_threshold}) {
        @val_based_entropies_to_be_returned = 
            @{$partitioning_point_child_entropies_hash{$best_feature_name}{$threshold_for_best_feature}};
    } else {
        @val_based_entropies_to_be_returned = ();
    }
    if (exists $partitioning_point_threshold{$best_feature_name}) {
        $decision_val_to_be_returned = $partitioning_point_threshold{$best_feature_name};
    } else {
        $decision_val_to_be_returned = undef;
    }
    print "\nBFC6 Val based entropies to be returned for feature $best_feature_name are " .
        "@val_based_entropies_to_be_returned\n"  if $self->{_debug3};
    return ($best_feature_name, $best_feature_entropy, \@val_based_entropies_to_be_returned, 
                                                                      $decision_val_to_be_returned);
}

sub number_of_nodes_created {
    DTNode->how_many_nodes();
}


#########################################    Entropy Calculators     #####################################

sub class_entropy_on_priors {
    my $self = shift;
    return $self->{_entropy_cache}->{'priors'} 
        if exists $self->{_entropy_cache}->{"priors"};
    my @class_names = @{$self->{_class_names}};
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
    $self->{_entropy_cache}->{'priors'} = $entropy;
    return $entropy;
}

sub entropy_scanner_for_a_numeric_feature {
    local $| = 1;
    my $self = shift;
    my $feature = shift;
    my @all_sampling_points = @{$self->{_sampling_points_for_numeric_feature_hash}->{$feature}};
    my @entropies_for_less_than_thresholds = ();
    my @entropies_for_greater_than_thresholds = ();
    foreach my $point (@all_sampling_points) {
        print ". ";
        push @entropies_for_less_than_thresholds, 
                         $self->class_entropy_for_less_than_threshold_for_feature([], $feature, $point);
        push @entropies_for_greater_than_thresholds,
                      $self->class_entropy_for_greater_than_threshold_for_feature([], $feature, $point);
    }
    print "\n\nSCANNER: All entropies less than thresholds for feature $feature are: ". 
                                                                "@entropies_for_less_than_thresholds\n";
    print "\nSCANNER: All entropies greater than thresholds for feature $feature are: ". 
                                                             "@entropies_for_greater_than_thresholds\n";
}   

sub class_entropy_for_less_than_threshold_for_feature {
    my $self = shift;
    my $arr = shift;
    my $feature = shift;
    my $threshold = shift;
    my @array_of_features_and_values_or_thresholds = @$arr;
    my $feature_threshold_combo = "$feature" . '<' . "$threshold";
    my $sequence = join ":", @array_of_features_and_values_or_thresholds;
    $sequence .= ":" . $feature_threshold_combo;
    return $self->{_entropy_cache}->{$sequence}  if exists $self->{_entropy_cache}->{$sequence};
    my @copy_of_array_of_features_and_values_or_thresholds = 
                                       @{deep_copy_array(\@array_of_features_and_values_or_thresholds)};
    push @copy_of_array_of_features_and_values_or_thresholds, $feature_threshold_combo;
    my $entropy = 0;
    foreach my $class_name (@{$self->{_class_names}}) {
        my $log_prob = undef;
        my $prob = $self->probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(
                                   $class_name, \@copy_of_array_of_features_and_values_or_thresholds);
        if ($prob >= 0.0001 && $prob <= 0.999) {
            $log_prob = log($prob) / log(2.0);
        } elsif ($prob < 0.0001) {
            $log_prob = 0;
        } elsif ($prob > 0.999) {
            $log_prob = 0;
        } else {
            die "An error has occurred in log_prob calculation";
        }
        $entropy +=  -1.0 * $prob * $log_prob;
    }
    if (abs($entropy) < 0.0000001) {
        $entropy = 0.0;
    }
    $self->{_entropy_cache}->{$sequence} = $entropy;
    return $entropy;
}

sub class_entropy_for_greater_than_threshold_for_feature {
    my $self = shift;
    my $arr = shift;
    my $feature = shift;
    my $threshold = shift;
    my @array_of_features_and_values_or_thresholds = @$arr;
    my $feature_threshold_combo = "$feature" . '>' . "$threshold";
    my $sequence = join ":", @array_of_features_and_values_or_thresholds;
    $sequence .= ":" . $feature_threshold_combo;
    return $self->{_entropy_cache}->{$sequence}  if exists $self->{_entropy_cache}->{$sequence};
    my @copy_of_array_of_features_and_values_or_thresholds = 
                                       @{deep_copy_array(\@array_of_features_and_values_or_thresholds)};
    push @copy_of_array_of_features_and_values_or_thresholds, $feature_threshold_combo;
    my $entropy = 0;
    foreach my $class_name (@{$self->{_class_names}}) {
        my $log_prob = undef;
        my $prob = $self->probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(
                                   $class_name, \@copy_of_array_of_features_and_values_or_thresholds);
        if ($prob >= 0.0001 && $prob <= 0.999) {
            $log_prob = log($prob) / log(2.0);
        } elsif ($prob < 0.0001) {
            $log_prob = 0;
        } elsif ($prob > 0.999) {
            $log_prob = 0;
        } else {
            die "An error has occurred in log_prob calculation";
        }
        $entropy +=  -1.0 * $prob * $log_prob;
    }
    if (abs($entropy) < 0.0000001) {
        $entropy = 0.0;
    }
    $self->{_entropy_cache}->{$sequence} = $entropy;
    return $entropy;
}

sub class_entropy_for_a_given_sequence_of_features_and_values_or_thresholds {
    my $self = shift;
    my $array_of_features_and_values_or_thresholds = shift;
    my @array_of_features_and_values_or_thresholds = @$array_of_features_and_values_or_thresholds;
    my $sequence = join ":", @array_of_features_and_values_or_thresholds;
    return $self->{_entropy_cache}->{$sequence}  if exists $self->{_entropy_cache}->{$sequence};
    my $entropy = 0;
    foreach my $class_name (@{$self->{_class_names}}) {
        my $log_prob = undef;
        my $prob = $self->probability_of_a_class_given_sequence_of_features_and_values_or_thresholds(
                                             $class_name, \@array_of_features_and_values_or_thresholds);
        if ($prob >= 0.0001 && $prob <= 0.999) {
            $log_prob = log($prob) / log(2.0);
        } elsif ($prob < 0.0001) {
            $log_prob = 0;
        } elsif ($prob > 0.999) {
            $log_prob = 0;
        } else {
            die "An error has occurred in log_prob calculation";
        }
        $entropy +=  -1.0 * $prob * $log_prob;
    }
    if (abs($entropy) < 0.0000001) {
        $entropy = 0.0;
    }
    $self->{_entropy_cache}->{$sequence} = $entropy;
    return $entropy;
}


#####################################   Probability Calculators   ########################################

sub prior_probability_for_class {
    my $self = shift;
    my $class = shift;
    my $class_name_in_cache = "prior" . '::' . $class;
    return $self->{_probability_cache}->{$class_name_in_cache}
        if exists $self->{_probability_cache}->{$class_name_in_cache};
    my $total_num_of_samples = keys %{$self->{_samples_class_label_hash}};
    my @values = values %{$self->{_samples_class_label_hash}};
    foreach my $class_name (@{$self->{_class_names}}) {
        my @trues = grep {$_ eq $class_name} @values;
        my $prior_for_this_class = (1.0 * @trues) / $total_num_of_samples; 
        my $this_class_name_in_cache = "prior" . '::' . $class_name;
        $self->{_probability_cache}->{$this_class_name_in_cache} = 
                                                  $prior_for_this_class;
    }
    return $self->{_probability_cache}->{$class_name_in_cache};
}

sub calculate_class_priors {
    my $self = shift;
    return if scalar keys %{$self->{_class_priors_hash}} > 1;
    foreach my $class_name (@{$self->{_class_names}}) {
        my $class_name_in_cache = "prior::$class_name";
        my $total_num_of_samples = scalar keys %{$self->{_samples_class_label_hash}};
        my @all_values = values %{$self->{_samples_class_label_hash}};
        my @trues_for_this_class = grep {$_ eq $class_name} @all_values;
        my $prior_for_this_class = (1.0 * (scalar @trues_for_this_class)) / $total_num_of_samples;
        $self->{_class_priors_hash}->{$class_name} = $prior_for_this_class;
        my $this_class_name_in_cache = "prior::$class_name";
        $self->{_probability_cache}->{$this_class_name_in_cache} = $prior_for_this_class;
    }
    if ($self->{_debug1}) {
        foreach my $class (sort keys %{$self->{_class_priors_hash}}) {
            print "$class  =>  $self->{_class_priors_hash}->{$class}\n";
        }
    }
}

sub calculate_first_order_probabilities {
    my $self = shift;
    foreach my $feature (@{$self->{_feature_names}}) {
        $self->probability_of_feature_value($feature, undef); 
        if ($self->{_debug2}) {
            if (exists $self->{_prob_distribution_numeric_features_hash}->{$feature}) {
                print "\nPresenting probability distribution for a numeric feature:\n";
                foreach my $sampling_point (sort {$a <=> $b} keys 
                                   %{$self->{_prob_distribution_numeric_features_hash}->{$feature}}) {
                    my $sampling_pt_for_display = sprintf("%.2f", $sampling_point);
                    print "$feature :: $sampling_pt_for_display=" . sprintf("%.5f", 
                          $self->{_prob_distribution_numeric_features_hash}->{$feature}{$sampling_point}) . "\n";
                }
            } else {
                print "\nPresenting probabilities for the values of a feature considered to be symbolic:\n";
                my @values_for_feature = @{$self->{_features_and_unique_values_hash}->{$feature}};
                foreach my $value (sort @values_for_feature) {
                    my $prob = $self->probability_of_feature_value($feature,$value); 
                    print "$feature :: $value = " . sprintf("%.5f", $prob) . "\n";
                }
            }
        }
    }
}

sub probability_of_feature_value {
    my $self = shift;
    my $feature_name = shift;
    my $value = shift;
    $value = sprintf("%.1f", $value) if defined($value) && $value =~ /^\d+$/;
    if (defined($value) && exists($self->{_sampling_points_for_numeric_feature_hash}->{$feature_name})) {
        $value = closest_sampling_point($value, 
                                        $self->{_sampling_points_for_numeric_feature_hash}->{$feature_name});
    }
    my $feature_and_value;
    if (defined($value)) {
        $feature_and_value = "$feature_name=$value";
    }
    if (defined($value) && exists($self->{_probability_cache}->{$feature_and_value})) {
        return $self->{_probability_cache}->{$feature_and_value};
    }
    my ($histogram_delta, $num_of_histogram_bins, @valuerange, $diffrange) = (undef,undef,undef,undef);
    if (exists $self->{_numeric_features_valuerange_hash}->{$feature_name}) {
        if ($self->{_feature_values_how_many_uniques_hash}->{$feature_name} > 
                                $self->{_symbolic_to_numeric_cardinality_threshold}) {
            if (! exists $self->{_sampling_points_for_numeric_feature_hash}->{$feature_name}) {
                @valuerange = @{$self->{_numeric_features_valuerange_hash}->{$feature_name}}; 
                $diffrange = $valuerange[1] - $valuerange[0];
                my %seen = ();
                my @unique_values_for_feature =  sort {$a <=> $b}  grep {$_ if $_ ne 'NA' && !$seen{$_}++} 
                                         @{$self->{_features_and_values_hash}->{$feature_name}};
                my @diffs = sort {$a <=> $b} map {$unique_values_for_feature[$_] - 
                                    $unique_values_for_feature[$_-1]}  1..@unique_values_for_feature-1;
                my $median_diff = $diffs[int(@diffs/2) - 1];
                $histogram_delta =  $median_diff * 2;
                if ($histogram_delta < $diffrange / 500.0) {
                    if (defined $self->{_number_of_histogram_bins}) {
                        $histogram_delta = $diffrange / $self->{_number_of_histogram_bins};
                    } else {
                        $histogram_delta = $diffrange / 500.0;
                    }
                }
                $self->{_histogram_delta_hash}->{$feature_name} = $histogram_delta;
                $num_of_histogram_bins = int($diffrange / $histogram_delta) + 1;
                $self->{_num_of_histogram_bins_hash}->{$feature_name} = $num_of_histogram_bins;
                my @sampling_points_for_feature = map {$valuerange[0] + $histogram_delta * $_} 
                                                                    0..$num_of_histogram_bins-1;
                $self->{_sampling_points_for_numeric_feature_hash}->{$feature_name} = 
                                                                           \@sampling_points_for_feature;
            }
        }
    }
    if (exists $self->{_numeric_features_valuerange_hash}->{$feature_name}) {
        if ($self->{_feature_values_how_many_uniques_hash}->{$feature_name} > 
                                   $self->{_symbolic_to_numeric_cardinality_threshold}) {
            my @sampling_points_for_feature = 
                               @{$self->{_sampling_points_for_numeric_feature_hash}->{$feature_name}};
            my @counts_at_sampling_points = (0) x @sampling_points_for_feature;
            my @actual_values_for_feature = grep {$_ ne 'NA'} 
                                              @{$self->{_features_and_values_hash}->{$feature_name}};
            foreach my $i (0..@sampling_points_for_feature-1) {
                foreach my $j (0..@actual_values_for_feature-1) {
                    if (abs($sampling_points_for_feature[$i]-$actual_values_for_feature[$j]) < $histogram_delta) {
                        $counts_at_sampling_points[$i]++
                    }
                }
            }
            my $total_counts = 0;
            map {$total_counts += $_} @counts_at_sampling_points;
            my @probs = map {$_ / (1.0 * $total_counts)} @counts_at_sampling_points;
            my %bin_prob_hash = ();
            foreach my $i (0..@sampling_points_for_feature-1) {
                $bin_prob_hash{$sampling_points_for_feature[$i]} = $probs[$i];
            }
            $self->{_prob_distribution_numeric_features_hash}->{$feature_name} = \%bin_prob_hash;
            my @values_for_feature = map "$feature_name=$_", map {sprintf("%.5f", $_)} 
                                                                    @sampling_points_for_feature;
            foreach my $i (0..@values_for_feature-1) {
                $self->{_probability_cache}->{$values_for_feature[$i]} = $probs[$i];
            }
            if (defined($value) && exists $self->{_probability_cache}->{$feature_and_value}) {
                return $self->{_probability_cache}->{$feature_and_value};
            } else {
                return 0;
            }
        } else {
            # This section is for numeric features that will be treated symbolically
            my %seen = ();
            my @values_for_feature = grep {$_ if $_ ne 'NA' && !$seen{$_}++} 
                                                 @{$self->{_features_and_values_hash}->{$feature_name}};
            @values_for_feature = map {"$feature_name=$_"} @values_for_feature;
            my @value_counts = (0) x @values_for_feature;

            foreach my $sample (sort {sample_index($a) cmp sample_index($b)} keys $self->{_training_data_hash}) {
                my @features_and_values = @{$self->{_training_data_hash}->{$sample}};
                foreach my $i (0..@values_for_feature-1) {
                    foreach my $current_value (@features_and_values) {
                        $value_counts[$i]++ if $values_for_feature[$i] eq $current_value;
                    }
                }
            }
            my $total_counts = 0;
            map {$total_counts += $_} @value_counts;
            die "PFV Something is wrong with your training file. It contains no training samples \
                         for feature named $feature_name" if $total_counts == 0;
            my @probs = map {$_ / (1.0 * $total_counts)} @value_counts;
            foreach my $i (0..@values_for_feature-1) {
                $self->{_probability_cache}->{$values_for_feature[$i]} = $probs[$i];
            }
            if (defined($value) && exists $self->{_probability_cache}->{$feature_and_value}) {
                return $self->{_probability_cache}->{$feature_and_value};
            } else {
                return 0;
            }
        }
    } else {
        # This section is only for purely symbolic features:  
        my @values_for_feature = @{$self->{_features_and_values_hash}->{$feature_name}};        
        @values_for_feature = map {"$feature_name=$_"} @values_for_feature;
        my @value_counts = (0) x @values_for_feature;
        foreach my $sample (sort {sample_index($a) cmp sample_index($b)} keys %{$self->{_training_data_hash}}) {
            my @features_and_values = @{$self->{_training_data_hash}->{$sample}};
            foreach my $i (0..@values_for_feature-1) {
                for my $current_value (@features_and_values) {
                    $value_counts[$i]++ if $values_for_feature[$i] eq $current_value;
                }
            }
        }
        foreach my $i (0..@values_for_feature-1) {
            $self->{_probability_cache}->{$values_for_feature[$i]} = 
                $value_counts[$i] / (1.0 * scalar(keys %{$self->{_training_data_hash}}));
        }
        if (defined($value) && exists $self->{_probability_cache}->{$feature_and_value}) {
            return $self->{_probability_cache}->{$feature_and_value};
        } else {
            return 0;
        }
    }
}

sub probability_of_feature_value_given_class {
    my $self = shift;
    my $feature_name = shift;
    my $feature_value = shift;
    my $class_name = shift;
    $feature_value = sprintf("%.1f", $feature_value) if defined($feature_value) && $feature_value =~ /^\d+$/;
    if (defined($feature_value) && exists($self->{_sampling_points_for_numeric_feature_hash}->{$feature_name})) {
        $feature_value = closest_sampling_point($feature_value, 
                                        $self->{_sampling_points_for_numeric_feature_hash}->{$feature_name});
    }
    my $feature_value_class;
    if (defined($feature_value)) {
        $feature_value_class = "$feature_name=$feature_value" . "::" . "$class_name";
    }
    if (defined($feature_value) && exists($self->{_probability_cache}->{$feature_value_class})) {
        print "\nNext answer returned by cache for feature $feature_name and " .
            "value $feature_value given class $class_name\n" if $self->{_debug2};
        return $self->{_probability_cache}->{$feature_value_class};
    }
    my ($histogram_delta, $num_of_histogram_bins, @valuerange, $diffrange) = (undef,undef,undef,undef);

    if (exists $self->{_numeric_features_valuerange_hash}->{$feature_name}) {
        if ($self->{_feature_values_how_many_uniques_hash}->{$feature_name} > 
                                $self->{_symbolic_to_numeric_cardinality_threshold}) {
            $histogram_delta = $self->{_histogram_delta_hash}->{$feature_name};
            $num_of_histogram_bins = $self->{_num_of_histogram_bins_hash}->{$feature_name};
            @valuerange = @{$self->{_numeric_features_valuerange_hash}->{$feature_name}};
            $diffrange = $valuerange[1] - $valuerange[0];
        }
    }
    my @samples_for_class = ();
    # Accumulate all samples names for the given class:
    foreach my $sample_name (keys %{$self->{_samples_class_label_hash}}) {
        if ($self->{_samples_class_label_hash}->{$sample_name} eq $class_name) {
            push @samples_for_class, $sample_name;
        }
    }
    if (exists($self->{_numeric_features_valuerange_hash}->{$feature_name})) {
        if ($self->{_feature_values_how_many_uniques_hash}->{$feature_name} > 
                                $self->{_symbolic_to_numeric_cardinality_threshold}) {
            my @sampling_points_for_feature = 
                              @{$self->{_sampling_points_for_numeric_feature_hash}->{$feature_name}};
            my @counts_at_sampling_points = (0) x @sampling_points_for_feature;
            my @actual_feature_values_for_samples_in_class = ();
            foreach my $sample (@samples_for_class) {           
                foreach my $feature_and_value (@{$self->{_training_data_hash}->{$sample}}) {
                    my $pattern = '(.+)=(.+)';
                    $feature_and_value =~ /$pattern/;
                    my ($feature, $value) = ($1, $2);
                    if (($feature eq $feature_name) && ($value ne 'NA')) {
                        push @actual_feature_values_for_samples_in_class, $value;
                    }
                }
            }
            foreach my $i (0..@sampling_points_for_feature-1) {
                foreach my $j (0..@actual_feature_values_for_samples_in_class-1) {
                    if (abs($sampling_points_for_feature[$i] - 
                            $actual_feature_values_for_samples_in_class[$j]) < $histogram_delta) {
                        $counts_at_sampling_points[$i]++;
                    }
                }
            }
            my $total_counts = 0;
            map {$total_counts += $_} @counts_at_sampling_points;
            die "PFVC1 Something is wrong with your training file. It contains no training " .
                    "samples for Class $class_name and Feature $feature_name" if $total_counts == 0;
            my @probs = map {$_ / (1.0 * $total_counts)} @counts_at_sampling_points;
            my @values_for_feature_and_class = map {"$feature_name=$_" . "::" . "$class_name"} 
                                                                     @sampling_points_for_feature;
            foreach my $i (0..@values_for_feature_and_class-1) {
                $self->{_probability_cache}->{$values_for_feature_and_class[$i]} = $probs[$i];
            }
            if (exists $self->{_probability_cache}->{$feature_value_class}) {
                return $self->{_probability_cache}->{$feature_value_class};
            } else {
                return 0;
            }
        } else {
            # This section is for numeric features that will be treated symbolically
            my %seen = ();
            my @values_for_feature = grep {$_ if $_ ne 'NA' && !$seen{$_}++} 
                                                 @{$self->{_features_and_values_hash}->{$feature_name}};
            @values_for_feature = map {"$feature_name=$_"} @values_for_feature;
            my @value_counts = (0) x @values_for_feature;
            foreach my $sample (@samples_for_class) {
                my @features_and_values = @{$self->{_training_data_hash}->{$sample}};
                foreach my $i (0..@values_for_feature-1) {
                    foreach my $current_value (@features_and_values) {
                        $value_counts[$i]++ if $values_for_feature[$i] eq $current_value;
                    }
                }
            }
            my $total_counts = 0;
            map {$total_counts += $_} @value_counts;
            die "PFVC2 Something is wrong with your training file. It contains no training " .
                "samples for Class $class_name and Feature $feature_name" if $total_counts == 0;
            # We normalize by total_count because the probabilities are conditioned on a given class
            foreach my $i (0..@values_for_feature-1) {
                my $feature_and_value_and_class =  "$values_for_feature[$i]" . "::" . "$class_name";
                $self->{_probability_cache}->{$feature_and_value_and_class} = 
                                                           $value_counts[$i] / (1.0 * $total_counts);
            }
            if (exists $self->{_probability_cache}->{$feature_value_class}) {
                return $self->{_probability_cache}->{$feature_value_class};
            } else {
                return 0;
            }
        }
    } else {
        # This section is for purely symbolic features
        my @values_for_feature = @{$self->{_features_and_values_hash}->{$feature_name}};
        my %seen = ();
        @values_for_feature = grep {$_ if $_ ne 'NA' && !$seen{$_}++} 
                                             @{$self->{_features_and_values_hash}->{$feature_name}};
        @values_for_feature = map {"$feature_name=$_"} @values_for_feature;
        my @value_counts = (0) x @values_for_feature;
        foreach my $sample (@samples_for_class) {
            my @features_and_values = @{$self->{_training_data_hash}->{$sample}};
            foreach my $i (0..@values_for_feature-1) {
                foreach my $current_value (@features_and_values) {
                    $value_counts[$i]++ if $values_for_feature[$i] eq $current_value;
                }
            }
        }
        my $total_counts = 0;
        map {$total_counts += $_} @value_counts;
        die "PFVC2 Something is wrong with your training file. It contains no training " .
            "samples for Class $class_name and Feature $feature_name" if $total_counts == 0;
        # We normalize by total_count because the probabilities are conditioned on a given class
        foreach my $i (0..@values_for_feature-1) {
            my $feature_and_value_and_class =  "$values_for_feature[$i]" . "::" . "$class_name";
            $self->{_probability_cache}->{$feature_and_value_and_class} = 
                                                       $value_counts[$i] / (1.0 * $total_counts);
        }
        if (exists $self->{_probability_cache}->{$feature_value_class}) {
            return $self->{_probability_cache}->{$feature_value_class};
        } else {
            return 0;
        }
    }
}

sub probability_of_feature_less_than_threshold {
    my $self = shift;
    my $feature_name = shift;
    my $threshold = shift;
    my $feature_threshold_combo = "$feature_name" . '<' . "$threshold";
    return $self->{_probability_cache}->{$feature_threshold_combo}
                     if (exists $self->{_probability_cache}->{$feature_threshold_combo});
    my @all_values = grep {$_ if $_ ne 'NA'} @{$self->{_features_and_values_hash}->{$feature_name}};
    my @all_values_less_than_threshold = grep {$_ if $_ <= $threshold} @all_values;
    my $probability = (1.0 * @all_values_less_than_threshold) / @all_values;
    $self->{_probability_cache}->{$feature_threshold_combo} = $probability;
    return $probability;
}

sub probability_of_feature_less_than_threshold_given_class {
    my $self = shift;
    my $feature_name = shift;
    my $threshold = shift;
    my $class_name = shift;
    my $feature_threshold_class_combo = "$feature_name" . "<" . "$threshold" . "::" . "$class_name";
    return $self->{_probability_cache}->{$feature_threshold_class_combo}
                     if (exists $self->{_probability_cache}->{$feature_threshold_class_combo});
    my @data_samples_for_class = ();
    # Accumulate all samples names for the given class:
    foreach my $sample_name (keys %{$self->{_samples_class_label_hash}}) {
        push @data_samples_for_class, $sample_name 
                  if $self->{_samples_class_label_hash}->{$sample_name} eq $class_name;
    }
    my @actual_feature_values_for_samples_in_class = ();
    foreach my $sample (@data_samples_for_class) {
        foreach my $feature_and_value (@{$self->{_training_data_hash}->{$sample}}) {
            my $pattern = '(.+)=(.+)';
            $feature_and_value =~ /$pattern/;
            my ($feature,$value) = ($1,$2);
            push @actual_feature_values_for_samples_in_class, $value
                                    if $feature eq $feature_name && $value ne 'NA';
        }
    }
    my @actual_points_for_feature_less_than_threshold = grep {$_ if $_ <= $threshold}
                                                    @actual_feature_values_for_samples_in_class;
    my $probability = (1.0 * @actual_points_for_feature_less_than_threshold) / 
                                                    @actual_feature_values_for_samples_in_class;
    $self->{_probability_cache}->{$feature_threshold_class_combo} = $probability;
    return $probability;
}

# This method requires that all truly numeric types only be expressed as '<' or '>'
# constructs in the array of branch features and thresholds
sub probability_of_a_sequence_of_features_and_values_or_thresholds {
    my $self = shift;
    my $arr = shift;
    my @array_of_features_and_values_or_thresholds = @$arr;
    return if scalar @array_of_features_and_values_or_thresholds == 0;
    my $sequence = join ':', @array_of_features_and_values_or_thresholds;
    return $self->{_probability_cache}->{$sequence} if exists $self->{_probability_cache}->{$sequence};
    my $probability = undef;
    my $pattern1 = '(.+)=(.+)';
    my $pattern2 = '(.+)<(.+)';
    my $pattern3 = '(.+)>(.+)';
    my @true_numeric_types = ();
    my @true_numeric_types_feature_names = ();
    my @symbolic_types = ();
    my @symbolic_types_feature_names = ();
    foreach my $item (@array_of_features_and_values_or_thresholds) {
        if ($item =~ /$pattern2/) {
            push @true_numeric_types, $item;
            my ($feature,$value) = ($1,$2);
            push @true_numeric_types_feature_names, $feature;
        } elsif ($item =~ /$pattern3/) {
            push @true_numeric_types, $item;
            my ($feature,$value) = ($1,$2);
            push @true_numeric_types_feature_names, $feature;
        } else {
            push @symbolic_types, $item;
            $item =~ /$pattern1/;
            my ($feature,$value) = ($1,$2);
            push @symbolic_types_feature_names, $feature;
        }
    }
    my %seen1 = ();
    @true_numeric_types_feature_names = grep {$_ if !$seen1{$_}++} @true_numeric_types_feature_names;
    my %seen2 = ();
    @symbolic_types_feature_names = grep {$_ if !$seen2{$_}++} @symbolic_types_feature_names;
    my $bounded_intervals_numeric_types = $self->find_bounded_intervals_for_numeric_features(\@true_numeric_types);
    print_array_with_msg("POS: Answer returned by find_bounded: ", 
                                       $bounded_intervals_numeric_types) if $self->{_debug2};
    # Calculate the upper and the lower bounds to be used when searching for the best
    # threshold for each of the numeric features that are in play at the current node:
    my (%upperbound, %lowerbound);
    foreach my $feature_name (@true_numeric_types_feature_names) {
        $upperbound{$feature_name} = undef;
        $lowerbound{$feature_name} = undef;
    }
    foreach my $item (@$bounded_intervals_numeric_types) {
        foreach my $feature_grouping (@$item) {
            if ($feature_grouping->[1] eq '>') {
                $lowerbound{$feature_grouping->[0]} = $feature_grouping->[2];
            } else {
                $upperbound{$feature_grouping->[0]} = $feature_grouping->[2];
            }
        }
    }
    foreach my $feature_name (@true_numeric_types_feature_names) {
        if (defined($lowerbound{$feature_name}) && defined($upperbound{$feature_name}) && 
                          $upperbound{$feature_name} <= $lowerbound{$feature_name}) { 
            return 0;
        } elsif (defined($lowerbound{$feature_name}) && defined($upperbound{$feature_name})) {
            if (! $probability) {
                $probability = $self->probability_of_feature_less_than_threshold($feature_name, 
                                                                                 $upperbound{$feature_name}) -
                   $self->probability_of_feature_less_than_threshold($feature_name, $lowerbound{$feature_name});
            } else {
                $probability *= ($self->probability_of_feature_less_than_threshold($feature_name, 
                                                                                   $upperbound{$feature_name}) -
                 $self->probability_of_feature_less_than_threshold($feature_name, $lowerbound{$feature_name}))
            }
        } elsif (defined($upperbound{$feature_name}) && ! defined($lowerbound{$feature_name})) {
            if (! $probability) {
                $probability = $self->probability_of_feature_less_than_threshold($feature_name,
                                                                                 $upperbound{$feature_name});
            } else {
                $probability *= $self->probability_of_feature_less_than_threshold($feature_name, 
                                                                                  $upperbound{$feature_name});
            }
        } elsif (defined($lowerbound{$feature_name}) && ! defined($upperbound{$feature_name})) {
            if (! $probability) {
                $probability = 1.0 - $self->probability_of_feature_less_than_threshold($feature_name,
                                                                                 $lowerbound{$feature_name});
            } else {
                $probability *= (1.0 - $self->probability_of_feature_less_than_threshold($feature_name, 
                                                                                $lowerbound{$feature_name}));
            }
        } else {
            die("Ill formatted call to 'probability_of_sequence' method");
        }
    }
    foreach my $feature_and_value (@symbolic_types) {
        if ($feature_and_value =~ /$pattern1/) {
            my ($feature,$value) = ($1,$2);
            if (! $probability) {        
                $probability = $self->probability_of_feature_value($feature, $value);
            } else {
                $probability *= $self->probability_of_feature_value($feature, $value);
            }
        }
    }
    $self->{_probability_cache}->{$sequence} = $probability;
    return $probability;
}

##  The following method requires that all truly numeric types only be expressed as
##  '<' or '>' constructs in the array of branch features and thresholds
sub probability_of_a_sequence_of_features_and_values_or_thresholds_given_class {
    my $self = shift;
    my $arr = shift;
    my $class_name = shift;
    my @array_of_features_and_values_or_thresholds = @$arr;
    return if scalar @array_of_features_and_values_or_thresholds == 0;
    my $sequence = join ':', @array_of_features_and_values_or_thresholds;
    my $sequence_with_class = "$sequence" . "::" . $class_name;
    return $self->{_probability_cache}->{$sequence_with_class} 
                      if exists $self->{_probability_cache}->{$sequence_with_class};
    my $probability = undef;
    my $pattern1 = '(.+)=(.+)';
    my $pattern2 = '(.+)<(.+)';
    my $pattern3 = '(.+)>(.+)';
    my @true_numeric_types = ();
    my @true_numeric_types_feature_names = ();
    my @symbolic_types = ();
    my @symbolic_types_feature_names = ();
    foreach my $item (@array_of_features_and_values_or_thresholds) {
        if ($item =~ /$pattern2/) {
            push @true_numeric_types, $item;
            my ($feature,$value) = ($1,$2);
            push @true_numeric_types_feature_names, $feature;
        } elsif ($item =~ /$pattern3/) {
            push @true_numeric_types, $item;
            my ($feature,$value) = ($1,$2);
            push @true_numeric_types_feature_names, $feature;
        } else {
            push @symbolic_types, $item;
            $item =~ /$pattern1/;
            my ($feature,$value) = ($1,$2);
            push @symbolic_types_feature_names, $feature;
        }
    }
    my %seen1 = ();
    @true_numeric_types_feature_names = grep {$_ if !$seen1{$_}++} @true_numeric_types_feature_names;
    my %seen2 = ();
    @symbolic_types_feature_names = grep {$_ if !$seen2{$_}++} @symbolic_types_feature_names;
    my $bounded_intervals_numeric_types = $self->find_bounded_intervals_for_numeric_features(\@true_numeric_types);
    print_array_with_msg("POSC: Answer returned by find_bounded: ", 
                                       $bounded_intervals_numeric_types) if $self->{_debug2};
    # Calculate the upper and the lower bounds to be used when searching for the best
    # threshold for each of the numeric features that are in play at the current node:
    my (%upperbound, %lowerbound);
    foreach my $feature_name (@true_numeric_types_feature_names) {
        $upperbound{$feature_name} = undef;
        $lowerbound{$feature_name} = undef;
    }
    foreach my $item (@$bounded_intervals_numeric_types) {
        foreach my $feature_grouping (@$item) {
            if ($feature_grouping->[1] eq '>') {
                $lowerbound{$feature_grouping->[0]} = $feature_grouping->[2];
            } else {
                $upperbound{$feature_grouping->[0]} = $feature_grouping->[2];
            }
        }
    }
    foreach my $feature_name (@true_numeric_types_feature_names) {
        if ($lowerbound{$feature_name} && $upperbound{$feature_name} && 
                          $upperbound{$feature_name} <= $lowerbound{$feature_name}) { 
            return 0;
        } elsif (defined($lowerbound{$feature_name}) && defined($upperbound{$feature_name})) {
            if (! $probability) {

                $probability =   $self->probability_of_feature_less_than_threshold_given_class($feature_name, 
                                                               $upperbound{$feature_name}, $class_name) -
                                 $self->probability_of_feature_less_than_threshold_given_class($feature_name, 
                                                               $lowerbound{$feature_name}, $class_name);
            } else {
                $probability *= ($self->probability_of_feature_less_than_threshold_given_class($feature_name, 
                                                               $upperbound{$feature_name}, $class_name) -
                                 $self->probability_of_feature_less_than_threshold_given_class($feature_name, 
                                                               $lowerbound{$feature_name}, $class_name))
            }
        } elsif (defined($upperbound{$feature_name}) && ! defined($lowerbound{$feature_name})) {
            if (! $probability) {
                $probability =   $self->probability_of_feature_less_than_threshold_given_class($feature_name,
                                                               $upperbound{$feature_name}, $class_name);
            } else {
                $probability *=  $self->probability_of_feature_less_than_threshold_given_class($feature_name, 
                                                               $upperbound{$feature_name}, $class_name);
            }
        } elsif (defined($lowerbound{$feature_name}) && ! defined($upperbound{$feature_name})) {
            if (! $probability) {
                $probability =   1.0 - $self->probability_of_feature_less_than_threshold_given_class($feature_name,
                                                               $lowerbound{$feature_name}, $class_name);
            } else {
                $probability *= (1.0 - $self->probability_of_feature_less_than_threshold_given_class($feature_name,
                                                               $lowerbound{$feature_name}, $class_name));
            }
        } else {
            die("Ill formatted call to 'probability of sequence given class' method");
        }
    }
    foreach my $feature_and_value (@symbolic_types) {
        if ($feature_and_value =~ /$pattern1/) {
            my ($feature,$value) = ($1,$2);
            if (! $probability) {        
                $probability = $self->probability_of_feature_value_given_class($feature, $value, $class_name);
            } else {
                $probability *= $self->probability_of_feature_value_given_class($feature, $value, $class_name);
            }
        }
    }
    $self->{_probability_cache}->{$sequence_with_class} = $probability;
    return $probability;
}

sub probability_of_a_class_given_sequence_of_features_and_values_or_thresholds {
    my $self = shift;
    my $class_name = shift;    
    my $arr = shift;
    my @array_of_features_and_values_or_thresholds = @$arr;
    my $sequence = join ':', @array_of_features_and_values_or_thresholds;
    my $class_and_sequence = "$class_name" . "::" . $sequence;
    return $self->{_probability_cache}->{$class_and_sequence} 
                      if exists $self->{_probability_cache}->{$class_and_sequence};
    my @array_of_class_probabilities = (0) x scalar @{$self->{_class_names}};
    foreach my $i (0..@{$self->{_class_names}}-1) {
        my $class_name = $self->{_class_names}->[$i];
        my $prob = $self->probability_of_a_sequence_of_features_and_values_or_thresholds_given_class(
                                               \@array_of_features_and_values_or_thresholds, $class_name);
        if ($prob < 0.000001) {
            $array_of_class_probabilities[$i] = 0.0;
            next;
        }
        my $prob_of_feature_sequence = $self->probability_of_a_sequence_of_features_and_values_or_thresholds(
                                                            \@array_of_features_and_values_or_thresholds);
#        die "PCS Something is wrong with your sequence of feature values and thresholds in " .
#                "probability_of_a_class_given_sequence_of_features_and_values_or_thresholds()"
#                if ! $prob_of_feature_sequence;
        my $prior = $self->{_class_priors_hash}->{$self->{_class_names}->[$i]};
        if ($prob_of_feature_sequence) {
            $array_of_class_probabilities[$i] = $prob * $prior / $prob_of_feature_sequence;
        } else {
            $array_of_class_probabilities[$i] =  $prior;
        }
    }
    my $sum_probability;
    map {$sum_probability += $_} @array_of_class_probabilities;
    if ($sum_probability == 0) {
        @array_of_class_probabilities =  map {1.0 / (scalar @{$self->{_class_names}})}  
                                                               (0..@{$self->{_class_names}}-1);
    } else {
        @array_of_class_probabilities = map {$_ * 1.0 / $sum_probability} @array_of_class_probabilities;
    }
    foreach my $i (0..@{$self->{_class_names}}-1) {
        my $this_class_and_sequence = "$self->{_class_names}->[$i]" . "::" . "$sequence";
        $self->{_probability_cache}->{$this_class_and_sequence} = $array_of_class_probabilities[$i];
    }
    return $self->{_probability_cache}->{$class_and_sequence};
}

#######################################  Class Based Utilities  ##########################################

##  Given a list of branch attributes for the numeric features of the form, say,
##  ['g2<1','g2<2','g2<3','age>34','age>36','age>37'], this method returns the
##  smallest list that is relevant for the purpose of calculating the probabilities.
##  To explain, the probability that the feature `g2' is less than 1 AND, at the same
##  time, less than 2, AND, at the same time, less than 3, is the same as the
##  probability that the feature less than 1. Similarly, the probability that 'age'
##  is greater than 34 and also greater than 37 is the same as `age' being greater
##  than 37.
sub find_bounded_intervals_for_numeric_features {
    my $self = shift;
    my $arr = shift;    
    my @arr = @$arr;
    my @features = @{$self->{_feature_names}};
    my @arr1 = map {my @x = split /(>|<)/, $_; \@x} @arr;   
    print_array_with_msg("arr1", \@arr1) if $self->{_debug2};
    my @arr3 = ();
    foreach my $feature_name (@features) {
        my @temp = ();
        foreach my $x (@arr1) {
            push @temp, $x if @$x > 0 && $x->[0] eq $feature_name;
        }
        push @arr3, \@temp if @temp > 0;
    }
    print_array_with_msg("arr3", \@arr3) if $self->{_debug2};
    # Sort each list so that '<' entries occur before '>' entries:
    my @arr4;
    foreach my $li (@arr3) {
        my @sorted = sort {$a->[1] cmp $b->[1]} @$li;
        push @arr4, \@sorted;
    }
    print_array_with_msg("arr4", \@arr4) if $self->{_debug2};
    my @arr5;
    foreach my $li (@arr4) {
        my @temp1 = ();
        my @temp2 = ();
        foreach my $inner (@$li) {
            if ($inner->[1] eq '<') {
                push @temp1, $inner;
            } else {
                push @temp2, $inner;
            }
        }
        if (@temp1 > 0 && @temp2 > 0) {
            push @arr5, [\@temp1, \@temp2];
        } elsif (@temp1 > 0) {
            push @arr5, [\@temp1];
        } else {
            push @arr5, [\@temp2];
        }
    }
    print_array_with_msg("arr5", \@arr5) if $self->{_debug2};
    my @arr6 = ();
    foreach my $li (@arr5) {
        my @temp1 = ();
        foreach my $inner (@$li) {
            my @sorted = sort {$a->[2] <=> $b->[2]} @$inner;
            push @temp1, \@sorted;
        }
        push @arr6, \@temp1;
    }
    print_array_with_msg("arr6", \@arr6) if $self->{_debug2};
    my @arr9 = ();
    foreach my $li (@arr6) {
        foreach my $alist (@$li) {
            my @newalist = ();
            if ($alist->[0][1] eq '<') {
                push @newalist, $alist->[0];
            } else {
                push @newalist, $alist->[-1];
            }
            if ($alist->[0][1] ne $alist->[-1][1]) {
                push @newalist, $alist->[-1];
            }
            push @arr9, \@newalist;
        }
    }
    print_array_with_msg('arr9', \@arr9) if $self->{_debug2};
    return \@arr9;

}

##  This method is used to verify that you used legal feature names in the test
##  sample that you want to classify with the decision tree.
sub check_names_used {
    my $self = shift;
    my $features_and_values_test_data = shift;
    my @features_and_values_test_data = @$features_and_values_test_data;
    my $pattern = '(\S+)\s*=\s*(\S+)';
    foreach my $feature_and_value (@features_and_values_test_data) {
        $feature_and_value =~ /$pattern/;
        my ($feature,$value) = ($1,$2);
        die "Your test data has formatting error" unless defined($feature) && defined($value);
        return 0 unless contained_in($feature, @{$self->{_feature_names}});
    }
    return 1;
}

#######################################  Data Condition Calculator  ######################################

##  This method estimates the worst-case fan-out of the decision tree taking into
##  account the number of values (and therefore the number of branches emanating from
##  a node) for the symbolic features.
sub determine_data_condition {
    my $self = shift;
    my $num_of_features = scalar @{$self->{_feature_names}};
    my @values = ();
    my @number_of_values;
    foreach my $feature (keys %{$self->{_features_and_unique_values_hash}}) {  
        push @values, @{$self->{_features_and_unique_values_hash}->{$feature}}
            if ! contained_in($feature, keys %{$self->{_numeric_features_valuerange_hash}});
        push @number_of_values, scalar @values;
    }
    return if ! @values;
    print "Number of features: $num_of_features\n";
    my @minmax = minmax(\@number_of_values);
    my $max_num_values = $minmax[1];
    print "Largest number of values for symbolic features is: $max_num_values\n";
    my $estimated_number_of_nodes = $max_num_values ** $num_of_features;
    print "\nWORST CASE SCENARIO: The decision tree COULD have as many as $estimated_number_of_nodes " .
          "nodes. The exact number of nodes created depends critically on " .
          "the entropy_threshold used for node expansion (the default value " .
          "for this threshold is 0.01) and on the value set for max_depth_desired " .
          "for the depth of the tree.\n";
    if ($estimated_number_of_nodes > 10000) {
        print "\nTHIS IS WAY TOO MANY NODES. Consider using a relatively " .
              "large value for entropy_threshold and/or a small value for " .
              "for max_depth_desired to reduce the number of nodes created.\n";
        print "\nDo you wish to continue anyway? Enter 'y' for yes:  ";
        my $answer = <STDIN>;
        chomp $answer;
        while ( ($answer !~ /y(es)?/i) && ($answer !~ /n(o)?/i) ) {
            print "\nAnswer not recognized.  Let's try again. Enter 'y' or 'n': ";
            $answer = <STDIN>;
            chomp $answer;            
        }
        die unless $answer =~ /y(es)?/i;
    }
}


####################################  Read Training Data From File  ######################################

sub get_training_data_from_csv {
    my $self = shift;
    my $numregex =  '[+-]?\ *(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?';
    my $filename = $self->{_training_datafile} || die "you did not specify a file for the training data";
    open FILEIN, $filename or die "Unable to open $filename: $!";
    die("Aborted. get_training_data_csv() is only for CSV files") unless $filename =~ /\.csv$/;
    my $class_name_in_column = $self->{_csv_class_column_index} - 1;   # subtract 1 because first col has labels
    my @all_data =  <FILEIN>;
    my %data_hash = ();
    foreach my $record (@all_data) {
        chomp $record;
        my @fields =  map {$_ =~ s/^\s*|\s*$//; $_} split /,/, $record;
        my @fields_after_first = @fields[1..$#fields]; 
        $data_hash{$fields[0]} = \@fields_after_first;
    }
    die 'Aborted. The first row of CSV file must begin with "" and then list the feature names and class names'
                                      unless exists $data_hash{'""'};
    my @field_names = map {$_ =~ s/^\s*\"|\"\s*$//g;$_} @{$data_hash{'""'}};
    my $class_column_heading = $field_names[$class_name_in_column];
    my @feature_names = map {$field_names[$_-1]} @{$self->{_csv_columns_for_features}};
    $class_column_heading =~ s/^\s*\"|\"\s*$//g;
    my %class_for_sample_hash = ();
    my %feature_values_for_samples_hash = ();
    foreach my $key (keys %data_hash) {
        next if $key =~ /^\"\"$/;
        my $cleanedup = $key;
        $cleanedup =~ s/^\s*\"|\"\s*$//g;
        my $which_class = $data_hash{$key}[$class_name_in_column];
        $which_class  =~ s/^\s*\"|\"\s*$//g;
        $class_for_sample_hash{"sample_$cleanedup"} = "$class_column_heading=$which_class";
        my @features_and_values_list = ();
        foreach my $i (@{$self->{_csv_columns_for_features}}) {
            my $feature_column_header = $field_names[$i-1];
            my $feature_val = $data_hash{$key}->[$i-1];
            $feature_val  =~ s/^\s*\"|\"\s*$//g;
            $feature_val = sprintf("%.1f",$feature_val) if $feature_val =~ /^\d+$/;
            push @features_and_values_list,  "$feature_column_header=$feature_val";
        }
        $feature_values_for_samples_hash{"sample_" . $cleanedup} = \@features_and_values_list;
    }
    my %features_and_values_hash = ();
    foreach my $i (@{$self->{_csv_columns_for_features}}) {
        my $feature = $data_hash{'""'}[$i-1];
        $feature =~ s/^\s*\"|\"\s*$//g;
        my @feature_values = ();
        foreach my $key (keys %data_hash) {     
            next if $key =~ /^\"\"$/;
            my $feature_val = $data_hash{$key}[$i-1];
            $feature_val =~ s/^\s*\"|\"\s*$//g;
            $feature_val = sprintf("%.1f",$feature_val) if $feature_val =~ /^\d+$/;
            push @feature_values, $feature_val;
        }
        $features_and_values_hash{$feature} = \@feature_values;
    }
    my %seen = ();
    my @all_class_names = grep {$_ if !$seen{$_}++}  values %class_for_sample_hash;
    print "\n All class names: @all_class_names\n" if $self->{_debug3};
    my %numeric_features_valuerange_hash = ();
    my %feature_values_how_many_uniques_hash = ();
    my %features_and_unique_values_hash = ();
    foreach my $feature (keys %features_and_values_hash) {
        my %seen1 = ();
        my @unique_values_for_feature = sort grep {$_ if $_ ne 'NA' && !$seen1{$_}++} 
                                                   @{$features_and_values_hash{$feature}};
        $feature_values_how_many_uniques_hash{$feature} = scalar @unique_values_for_feature;
        my $not_all_values_float = 0;
        #map {$not_all_values_float = 1 if $_ !~ /^\d*\.\d+$/} @unique_values_for_feature;
        map {$not_all_values_float = 1 if $_ !~ /^$numregex$/} @unique_values_for_feature;
        if ($not_all_values_float == 0) {
            my @minmaxvalues = minmax(\@unique_values_for_feature);
            $numeric_features_valuerange_hash{$feature} = \@minmaxvalues; 
        }
        $features_and_unique_values_hash{$feature} = \@unique_values_for_feature;
    }
    if ($self->{_debug1}) {
        print "\nAll class names: @all_class_names\n";
        print "\nEach sample data record:\n";
        foreach my $sample (sort {sample_index($a) <=> sample_index($b)} keys %feature_values_for_samples_hash) {
            print "$sample  =>  @{$feature_values_for_samples_hash{$sample}}\n";
        }
        print "\nclass label for each data sample:\n";
        foreach my $sample (sort {sample_index($a) <=> sample_index($b)}  keys %class_for_sample_hash) {
            print "$sample => $class_for_sample_hash{$sample}\n";
        }
        print "\nFeatures used: @feature_names\n\n";
        print "\nfeatures and the values taken by them:\n";
        foreach my $feature (sort keys %features_and_values_hash) {
            print "$feature => @{$features_and_values_hash{$feature}}\n";
        }
        print "\nnumeric features and their ranges:\n";
        foreach  my $feature (sort keys %numeric_features_valuerange_hash) {
            print "$feature  =>  @{$numeric_features_valuerange_hash{$feature}}\n";
        }
        print "\nnumber of unique values in each feature:\n";
        foreach  my $feature (sort keys %feature_values_how_many_uniques_hash) {
            print "$feature  =>  $feature_values_how_many_uniques_hash{$feature}\n";
        }
    }
    $self->{_class_names} = \@all_class_names;
    $self->{_feature_names} = \@feature_names;
    $self->{_samples_class_label_hash}  =  \%class_for_sample_hash;
    $self->{_training_data_hash}  =  \%feature_values_for_samples_hash;
    $self->{_features_and_values_hash}  = \%features_and_values_hash;
    $self->{_features_and_unique_values_hash}  =  \%features_and_unique_values_hash;
    $self->{_numeric_features_valuerange_hash} = \%numeric_features_valuerange_hash;
    $self->{_feature_values_how_many_uniques_hash} = \%feature_values_how_many_uniques_hash;
}

##  If your training data is purely symbolic, as in version 1.7.1 or lower, you are
##  better off creating a `.dat' file.  For purely numeric data or mixed data, place
##  it in a `.csv' file.  See examples of these files in the `examples' subdirectory.
sub get_training_data {
    my $self = shift;
    my $filename = $self->{_training_datafile};
    $self->get_training_data_from_csv() if $filename =~ /.csv$/;
    $self->get_training_data_from_dat() if $filename =~ /.dat$/;
}

##  Meant for purely symbolic data (as in all versions up to v. 1.7.1)
sub get_training_data_from_dat {
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
    my @class_names;
    open INPUT, $training_data_file
                || "unable to open training data file: $!";
    my @all_data;
    my $all_data_as_string;
    my %training_data_records;
    while (<INPUT>) {
        chomp;
        next if /^[\s=]*$/;
        next if /^#.*$/;
        if ( (/^class/i) && !$recording_training_data 
                         && !$recording_features_flag ) {
            $_ =~ /^\s*class names:\s*([ \S]+)\s*/i;
            @class_names = split /\s+/, $1;
            @class_names = grep {defined($_) && length($_) > 0} @class_names;
            my @bad_names = grep /\W+/, @class_names;            
            die "Your class names near the top of the training file do not look clean.  Class names " .
                "are only allowed to have alphanumeric characters (including the underscore).  " .
                "They must not be separated by punctuation marks such as commas. The class names " .
                "must be separated by white space only" 
                                                  unless @bad_names == 0;
            die "You have not supplied any class names in your training file"
                if @class_names == 0;
            next;
        } elsif (/^feature names and their values/i) {
            $recording_features_flag = 1;
            next;
        } elsif (/^training data/i) {
            $recording_training_data = 1;
            $recording_features_flag = 0;
            next;
        } elsif ( !$recording_training_data && $recording_features_flag ) {
            my ($feature_name, $value_string) = $_ =~ /^\s*(\S+)\s*=\s*(.+)/i;
            $features_and_values_hash{$feature_name} = [];
            my @values = split /\s+/, $value_string;
            @values = grep {defined($_) && length($_) > 0} @values;
            push @{$features_and_values_hash{$feature_name}}, @values;
            push @feature_names, $feature_name;
            my %seen = ();
            my @unique_values = sort map {$_ if !$seen{$_}++} @values;
            $self->{_features_and_unique_values_hash}->{$feature_name} = \@unique_values;
        } elsif ($recording_training_data) {
            if (@table_header == 0) {
                @table_header = split;
                @table_header = grep {defined($_) && length($_) > 0} @table_header;
                foreach my $i (2..@table_header-1) {
                    $column_label_hash{$i} = $table_header[$i];
                }
                next;
            }
            my @record = split /\s+/;
            @record =  grep {defined($_) && length($_) > 0} @record;
            my $num_of_feature_vals = scalar(@record) - 2;
            my $num_of_features = @feature_names;
            die "Your training datafile is defective.  The number of values, $num_of_feature_vals, " .
                "declared for $record[0] does not match the number of features declared at the " .
                "beginning of the datafile. You previously declared  $num_of_features features." 
                                       unless @record == $num_of_features + 2;
            my $class_name_in_record = $record[1];
            die "The class name in sample record $record[0] does not match the class names declared " .
                "at the top of your training file"
                                  unless contained_in($class_name_in_record, @class_names);
            $samples_class_label_hash{$record[0]} = $record[1];
            $training_data_hash{$record[0]} = [];
            foreach my $i (2..@record-1) {
                my $feature_name_for_this_col = $column_label_hash{$i};
                my @values_for_this_feature = 
                    @{$features_and_values_hash{$feature_name_for_this_col}};
                die "For sample record $record[0], one of the feature values, $record[$i], is illegal. " .
                    "Legal values for this feature, $column_label_hash{$i}, are: @values_for_this_feature"
                                      unless contained_in($record[$i], @values_for_this_feature);
                push @{$training_data_hash{$record[0]}}, 
                                "$column_label_hash{$i}" . "=" . $record[$i];
            }
        }
    }
    my @class_labels_in_sample_records = values(%samples_class_label_hash);
    my @empty_classes;
    foreach my $classname (@class_names) {
        push @empty_classes, $classname
            unless contained_in($classname, @class_labels_in_sample_records) 
    }
    if (@empty_classes) {
        my $num_empty_classes = @empty_classes;       
        print "\nDid you know you have $num_empty_classes class(es).  The DecisionTree module " .
              "can ignore these classes for you.\n";
        print "EMPTY CLASSES: @empty_classes\n";
        print "Do you wish to continue? Enter 'y' if yes:  ";
        my $answer = <STDIN>;
        $answer =~ s/^\s*(\w+)\s*$/\L$1/;
        die "You chose to not continue with empty classes" 
            unless $answer eq 'y';
        my @new_class_names;
        foreach my $classname (@class_names) {
            push @new_class_names, $classname
                unless contained_in($classname, @empty_classes) 
        }
        @class_names = @new_class_names;
    }
    $self->{_class_names} = \@class_names;
    $self->{_feature_names} = \@feature_names;
    $self->{_features_and_values_hash} = \%features_and_values_hash;
    $self->{_samples_class_label_hash} = \%samples_class_label_hash;
    $self->{_training_data_hash} = \%training_data_hash;
    if ($self->{_debug1}) {
        print "\nClass names: @{$self->{_class_names}}\n";
        print "\nFeature names: @{$self->{_feature_names}}\n";
        print "Features and values:\n";
        foreach my $feature (sort keys %{$self->{_features_and_values_hash}}) {
            print "$feature  =>  @{$self->{_features_and_values_hash}->{$feature}}\n";
        }
    }
    foreach my $feature (@feature_names) {
        my @values_for_feature = @{$features_and_values_hash{$feature}};
        foreach my $value (@values_for_feature) {
            my $feature_and_value = "$feature=$value";
            $self->{_probability_cache}->{$feature_and_value} =  
                                            $self->probability_of_feature_value($feature,$value); 
        }
    }
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

##########################################  Utility Routines  ############################################

sub closest_sampling_point {
    my $value = shift;
    my $arr_ref = shift;
    my @arr = @{$arr_ref};
    my @compare = map {abs($_ - $value)} @arr;
    my ($minval,$index) = minimum(\@compare);
    return $arr[$index];
}

# returns the array index that contains a specified STRING value:
# meant only for array of strings
sub get_index_at_value {
    my $value = shift;
    my @array = @{shift @_};
    foreach my $i (0..@array-1) {
        return $i if $value eq $array[$i];
    }
}

##  When the training data is read from a CSV file, we assume that the first column
##  of each data record contains a unique integer identifier for the record in that
##  row. This training data is stored in a hash whose keys are the prefix 'sample_'
##  followed by the identifying integers.  For the data in the old-style `.dat'
##  files, we assume that each record begins with the string `sample_xx' where `xx'
##  is a unique integer.  In both cases, the purpose of this function is to return
##  the identifying integer associated with a data record.
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

# Returns an array of two values, the min and the max, of an array of floats
sub minmax {
    my $arr = shift;
    my ($min, $max);
    foreach my $i (0..@{$arr}-1) {
        if ( (!defined $min) || ($arr->[$i] < $min) ) {
            $min = $arr->[$i];
        }
        if ( (!defined $max) || ($arr->[$i] > $max) ) {
            $max = $arr->[$i];
        }
    }
    return ($min, $max);
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
    return [] if scalar @$ref_in == 0;
    foreach my $i (0..@{$ref_in}-1) {
        $ref_out->[$i] = $ref_in->[$i];
    }
    return $ref_out;
}

sub check_for_illegal_params2 {
    my @params = @_;
    my @legal_params = qw / training_datafile
                            entropy_threshold
                            max_depth_desired
                            csv_class_column_index
                            csv_columns_for_features
                            symbolic_to_numeric_cardinality_threshold
                            number_of_histogram_bins
                            debug1
                            debug2
                            debug3
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

sub print_array_with_msg {
    my $message = shift;
    my $arr = shift;
    print "\n$message: ";
    print_nested_array( $arr );
}

sub print_nested_array {
    my $arr = shift;
    my @arr = @$arr;
    print "[";
    foreach my $item (@arr) {
        if (ref $item) {
            print_nested_array($item);
        } else {
            print "$item";
        }
    }
    print "]";
}    

######################################### Class EvalTrainingData  ########################################

##  This subclass of the DecisionTree class is used to evaluate the quality of your
##  training data by running a 10-fold cross-validation test on it. This test divides
##  all of the training data into ten parts, with nine parts used for training a
##  decision tree and one part used for testing its ability to classify correctly.
##  This selection of nine parts for training and one part for testing is carried out
##  in all of the ten different possible ways.  This testing functionality can also
##  be used to find the best values to use for the constructor parameters
##  entropy_threshold, max_depth_desired, and
##  symbolic_to_numeric_cardinality_threshold.

##  Only the CSV training files can be evaluated in this manner (because only CSV
##  training are allowed to have numeric features --- which is the more interesting
##  case for evaluation analytics.

package EvalTrainingData;

@EvalTrainingData::ISA = ('Algorithm::DecisionTree');

sub new {
    my $class = shift;
    my $instance = Algorithm::DecisionTree->new(@_);
    bless $instance, $class;
}

sub evaluate_training_data {
    my $self = shift;
    my $evaldebug = 0;
    die "The data evaluation function in the module can only be used when your " .
        "training data is in a CSV file" unless $self->{_training_datafile} =~ /\.csv$/;
    print "\nWill run a 10-fold cross-validation test on your training data to test its " .
          "class-discriminatory power:\n";
    my %all_training_data = %{$self->{_training_data_hash}};
    my @all_sample_names = sort {Algorithm::DecisionTree::sample_index($a) <=> 
                                     Algorithm::DecisionTree::sample_index($b)}  keys %all_training_data;
    my $fold_size = int(0.1 * (scalar keys %all_training_data));
    print "fold size: $fold_size\n";
    my %confusion_matrix = ();
    foreach my $class_name (@{$self->{_class_names}}) {
        foreach my $inner_class_name (@{$self->{_class_names}}) {
            $confusion_matrix{$class_name}->{$inner_class_name} = 0;
        }
    }
    foreach my $fold_index (0..9) {
        print "\nStarting the iteration indexed $fold_index of the 10-fold cross-validation test\n"; 
        my @testing_samples = @all_sample_names[$fold_size * $fold_index .. $fold_size * ($fold_index+1) - 1];
        my @training_samples = (@all_sample_names[0 .. $fold_size * $fold_index-1],  
                     @all_sample_names[$fold_size * ($fold_index+1) .. (scalar keys %all_training_data) - 1]);
        my %testing_data = ();
        foreach my $x (@testing_samples) {
            $testing_data{$x} = $all_training_data{$x};
        }
        my %training_data = ();
        foreach my $x (@training_samples) {
            $training_data{$x} = $all_training_data{$x};
        }
        my $trainingDT = Algorithm::DecisionTree->new('evalmode');
        $trainingDT->{_training_data_hash} = \%training_data;
        $trainingDT->{_class_names} = $self->{_class_names};
        $trainingDT->{_feature_names} = $self->{_feature_names};
        $trainingDT->{_entropy_threshold} = $self->{_entropy_threshold};
        $trainingDT->{_max_depth_desired} = $self->{_max_depth_desired};
        $trainingDT->{_symbolic_to_numeric_cardinality_threshold} = 
                                                $self->{_symbolic_to_numeric_cardinality_threshold};
        foreach my $sample_name (@training_samples) {
            $trainingDT->{_samples_class_label_hash}->{$sample_name} = 
                                                $self->{_samples_class_label_hash}->{$sample_name};
        }
        foreach my $feature (keys %{$self->{_features_and_values_hash}}) {
            $trainingDT->{_features_and_values_hash}->{$feature} = ();
        }
        my $pattern = '(\S+)\s*=\s*(\S+)';
        foreach my $item (sort {Algorithm::DecisionTree::sample_index($a) <=> 
                                Algorithm::DecisionTree::sample_index($b)}  
                          keys %{$trainingDT->{_training_data_hash}}) {
            foreach my $feature_and_value (@{$trainingDT->{_training_data_hash}->{$item}}) {
                $feature_and_value =~ /$pattern/;
                my ($feature,$value) = ($1,$2);
                push @{$trainingDT->{_features_and_values_hash}->{$feature}}, $value if $value ne 'NA';
            }
        }
        foreach my $feature (keys %{$trainingDT->{_features_and_values_hash}}) {
            my %seen = ();
            my @unique_values_for_feature = grep {$_ if $_ ne 'NA' && !$seen{$_}++} 
                                                @{$trainingDT->{_features_and_values_hash}->{$feature}}; 
            if (Algorithm::DecisionTree::contained_in($feature, 
                                                keys %{$self->{_numeric_features_valuerange_hash}})) {
                @unique_values_for_feature = sort {$a <=> $b} @unique_values_for_feature;
            } else {
                @unique_values_for_feature = sort @unique_values_for_feature;
            }
            $trainingDT->{_features_and_unique_values_hash}->{$feature} = \@unique_values_for_feature;
        }
        foreach my $feature (keys %{$self->{_numeric_features_valuerange_hash}}) {
            my @minmaxvalues = Algorithm::DecisionTree::minmax(
                                         \@{$trainingDT->{_features_and_unique_values_hash}->{$feature}});
            $trainingDT->{_numeric_features_valuerange_hash}->{$feature} = \@minmaxvalues;
        }
        if ($evaldebug) {
            print "\n\nprinting samples in the testing set: @testing_samples\n";
            print "\n\nPrinting features and their values in the training set:\n";
            foreach my $item (sort keys %{$trainingDT->{_features_and_values_hash}}) {
                print "$item  => @{$trainingDT->{_features_and_values_hash}->{$item}}\n";
            }
            print "\n\nPrinting unique values for features:\n";
            foreach my $item (sort keys %{$trainingDT->{_features_and_unique_values_hash}}) {
                print "$item  => @{$trainingDT->{_features_and_unique_values_hash}->{$item}}\n";
            }
            print "\n\nPrinting unique value ranges for features:\n";
            foreach my $item (sort keys %{$trainingDT->{_numeric_features_valuerange_hash}}) {
                print "$item  => @{$trainingDT->{_numeric_features_valuerange_hash}->{$item}}\n";
            }
        }
        foreach my $feature (keys %{$self->{_features_and_unique_values_hash}}) {
            $trainingDT->{_feature_values_how_many_uniques_hash}->{$feature} = 
                scalar @{$trainingDT->{_features_and_unique_values_hash}->{$feature}};
        }
        $trainingDT->{_debug2} = 1 if $evaldebug;
        $trainingDT->calculate_first_order_probabilities();
        $trainingDT->calculate_class_priors();
        my $root_node = $trainingDT->construct_decision_tree_classifier();
        $root_node->display_decision_tree("     ") if $evaldebug;
        foreach my $test_sample_name (@testing_samples) {
            my @test_sample_data = @{$all_training_data{$test_sample_name}};
            print "original data in test sample: @test_sample_data\n" if $evaldebug;
            @test_sample_data = grep {$_ if $_ && $_ !~ /=NA$/} @test_sample_data;
            print "filtered data in test sample: @test_sample_data\n" if $evaldebug;
            my %classification = %{$trainingDT->classify($root_node, \@test_sample_data)};
            my @solution_path = @{$classification{'solution_path'}};
            delete $classification{'solution_path'};
            my @which_classes = keys %classification;
            @which_classes = sort {$classification{$b} <=> $classification{$a}} @which_classes;
            my $most_likely_class_label = $which_classes[0];
            if ($evaldebug) {
                print "\nClassification:\n\n";
                print "     class                         probability\n";
                print "     ----------                    -----------\n";
                foreach my $which_class (@which_classes) {
                    my $classstring = sprintf("%-30s", $which_class);
                    my $valuestring = sprintf("%-30s", $classification{$which_class});
                    print "     $classstring $valuestring\n";
                }
                print "\nSolution path in the decision tree: @solution_path\n";
                print "\nNumber of nodes created: " . $root_node->how_many_nodes() . "\n";
            }
            my $true_class_label_for_sample = $self->{_samples_class_label_hash}->{$test_sample_name};
            print "$test_sample_name:    true_class: $true_class_label_for_sample    " .
                     "estimated_class: $most_likely_class_label\n"  if $evaldebug;
            $confusion_matrix{$true_class_label_for_sample}->{$most_likely_class_label} += 1;
        }
    }
    print "\n\n       DISPLAYING THE CONFUSION MATRIX FOR THE 10-FOLD CROSS-VALIDATION TEST:\n\n\n";
    my $matrix_header = " " x 30;
    foreach my $class_name (@{$self->{_class_names}}) {  
        $matrix_header .= sprintf("%-30s", $class_name);
    }
    print "\n" . $matrix_header . "\n\n";
    foreach my $row_class_name (sort keys %confusion_matrix) {
        my $row_display = sprintf("%-30s", $row_class_name);
        foreach my $col_class_name (sort keys %{$confusion_matrix{$row_class_name}}) {
            $row_display .= sprintf( "%-30u",  $confusion_matrix{$row_class_name}->{$col_class_name} );
        }
        print "$row_display\n\n";
    }
    print "\n\n";
    my ($diagonal_sum, $off_diagonal_sum) = (0,0);
    foreach my $row_class_name (sort keys %confusion_matrix) {
        foreach my $col_class_name (sort keys %{$confusion_matrix{$row_class_name}}) {
            if ($row_class_name eq $col_class_name) {
                $diagonal_sum += $confusion_matrix{$row_class_name}->{$col_class_name};
            } else {
                $off_diagonal_sum += $confusion_matrix{$row_class_name}->{$col_class_name};
            }
        }
    }
    my $data_quality_index = 100.0 * $diagonal_sum / ($diagonal_sum + $off_diagonal_sum);
    print "\nTraining Data Quality Index: $data_quality_index    (out of a possible maximum of 100)\n";
    if ($data_quality_index <= 80) {
        print "\nYour training data does not possess much class discriminatory " .
              "information.  It could be that the classes are inherently not well " .
              "separable or that your constructor parameter choices are not appropriate.\n";
    } elsif ($data_quality_index > 80 && $data_quality_index <= 90) {
        print "\nYour training data possesses some class discriminatory information " .
              "but it may not be sufficient for real-world applications.  You might " .
              "try tweaking the constructor parameters to see if that improves the " .
              "class discriminations.\n";
    } elsif ($data_quality_index > 90 && $data_quality_index <= 95) {
        print  "\nYour training data appears to possess good class discriminatory " .
               "information.  Whether or not it is acceptable would depend on your " .
               "application.\n";
    } elsif ($data_quality_index > 95 && $data_quality_index <= 98) {
        print "\nYour training data is of excellent quality.\n";
    } else {
        print "\nYour training data is perfect.\n";
    }

}


#############################################  Class DTNode  #############################################

# The nodes of the decision tree are instances of this class:

package DTNode;

use strict; 
use Carp;

our $nodes_created = 0;
our @class_names;

# $feature is the feature test at the current node.  $branch_features_and_values is
# an anonymous array holding the feature names and corresponding values on the path
# from the root to the current node:
sub new {                                                           
    my ($class, $feature, $entropy, $class_probabilities, 
                                       $branch_features_and_values_or_thresholds, $root_or_not) = @_; 
    DTNode::initialize_DTNode_class() if defined($root_or_not) && $root_or_not eq 'root';
    bless {                                                         
        _serial_number           => $nodes_created++,
        _feature                 => $feature,                                       
        _node_creation_entropy   => $entropy,
        _class_probabilities     => $class_probabilities,
        _branch_features_and_values_or_thresholds => $branch_features_and_values_or_thresholds,
        _linked_to => [],                                          
    }, $class;                                                     
}

sub initialize_DTNode_class {
    $DTNode::nodes_created = 0;
    @DTNode::class_names = undef;
}

sub how_many_nodes {
    my $self = shift;
    return $nodes_created;
}

# class method
sub set_class_names {
    my $class = shift;
    die "illegal invocation of a class method" 
        unless $class eq 'DTNode';
    my $class_names_list = shift;
    @class_names = @{$class_names_list};
}

# class method
sub get_class_names {
    my $class = shift;
    die "illegal invocation of a class method" 
        unless $class eq 'DTNode';
    return \@class_names;
}

sub get_serial_num {
    my $self = shift;
    $self->{_serial_number};
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

sub get_node_entropy {
    my $self = shift;                              
    return $self->{_node_creation_entropy};
}

sub get_class_probabilities {                                  
    my $self = shift;                              
    return $self->{ _class_probabilities};                    
}

sub get_branch_features_and_values_or_thresholds {
    my $self = shift; 
    return $self->{_branch_features_and_values_or_thresholds};
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

sub display_node {
    my $self = shift; 
    my $feature_at_node = $self->get_feature() || " ";
    my $node_creation_entropy_at_node = $self->get_node_entropy();
    my $print_node_creation_entropy_at_node = sprintf("%.3f", $node_creation_entropy_at_node);
    my @class_probabilities = @{$self->get_class_probabilities()};
    my @class_probabilities_for_display = map {sprintf("%0.3f", $_)} @class_probabilities;
    my $serial_num = $self->get_serial_num();
    my @branch_features_and_values_or_thresholds = @{$self->get_branch_features_and_values_or_thresholds()};
    print "\n\nNODE $serial_num" .
          ":\n   Branch features and values to this node: @branch_features_and_values_or_thresholds" .
          "\n   Class probabilities at current node: @class_probabilities_for_display" .
          "\n   Entropy at current node: $print_node_creation_entropy_at_node" .
          "\n   Best feature test at current node: $feature_at_node\n\n";
}

sub display_decision_tree {
    my $self = shift;
    my $offset = shift;
    my $serial_num = $self->get_serial_num();
    if (@{$self->get_children()} > 0) {
        my $feature_at_node = $self->get_feature() || " ";
        my $node_creation_entropy_at_node = $self->get_node_entropy();
        my $print_node_creation_entropy_at_node = sprintf("%.3f", $node_creation_entropy_at_node);
        my @branch_features_and_values_or_thresholds = @{$self->get_branch_features_and_values_or_thresholds()};
        my @class_probabilities = @{$self->get_class_probabilities()};
        my @print_class_probabilities = map {sprintf("%0.3f", $_)} @class_probabilities;
        my @class_names = @{DTNode->get_class_names()};
        my @print_class_probabilities_with_class =
            map {"$class_names[$_]" . '=>' . $print_class_probabilities[$_]} 0..@class_names-1;
        print "NODE $serial_num: $offset BRANCH TESTS TO NODE: @branch_features_and_values_or_thresholds\n";
        my $second_line_offset = "$offset" . " " x (8 + length("$serial_num"));
        print "$second_line_offset" . "Decision Feature: $feature_at_node    Node Creation Entropy: " ,
              "$print_node_creation_entropy_at_node   Class Probs: @print_class_probabilities_with_class\n\n";
        $offset .= "   ";
        foreach my $child (@{$self->get_children()}) {
            $child->display_decision_tree($offset);
        }
    } else {
        my $node_creation_entropy_at_node = $self->get_node_entropy();
        my $print_node_creation_entropy_at_node = sprintf("%.3f", $node_creation_entropy_at_node);
        my @branch_features_and_values_or_thresholds = @{$self->get_branch_features_and_values_or_thresholds()};
        my @class_probabilities = @{$self->get_class_probabilities()};
        my @print_class_probabilities = map {sprintf("%0.3f", $_)} @class_probabilities;
        my @class_names = @{DTNode->get_class_names()};
        my @print_class_probabilities_with_class =
            map {"$class_names[$_]" . '=>' . $print_class_probabilities[$_]} 0..@class_names-1;
        print "NODE $serial_num: $offset BRANCH TESTS TO LEAF NODE: @branch_features_and_values_or_thresholds\n";
        my $second_line_offset = "$offset" . " " x (8 + length("$serial_num"));
        print "$second_line_offset" . "Node Creation Entropy: $print_node_creation_entropy_at_node   " .
              "Class Probs: @print_class_probabilities_with_class\n\n";
    }
}


##########################  Generate Your Own Numeric Training and Test Data  ############################
##########################      Class TrainingAndTestDataGeneratorNumeric     ############################

##  See the script generate_training_and_test_data_numeric.pl in the examples
##  directory on how to use this class for generating your own numeric training and
##  test data.  The training and test data are generated in accordance with the
##  specifications you place in the parameter file that is supplied as an argument to
##  the constructor of this class.

package TrainingAndTestDataGeneratorNumeric;

use strict;                                                         
use Carp;

sub new {                                                           
    my ($class, %args) = @_;
    my @params = keys %args;
    croak "\nYou have used a wrong name for a keyword argument " .
          "--- perhaps a misspelling\n" 
          if check_for_illegal_params3(@params) == 0;   
    bless {
        _output_training_csv_file          =>   $args{'output_training_csv_file'} 
                                                   || croak("name for output_training_csv_file required"),
        _output_test_csv_file              =>   $args{'output_test_csv_file'} 
                                                   || croak("name for output_test_csv_file required"),
        _parameter_file                    =>   $args{'parameter_file'}
                                                         || croak("parameter_file required"),
        _number_of_samples_for_training    =>   $args{'number_of_samples_for_training'} 
                                                         || croak("number_of_samples_for_training"),
        _number_of_samples_for_testing     =>   $args{'number_of_samples_for_testing'} 
                                                         || croak("number_of_samples_for_testing"),
        _debug                             =>    $args{debug} || 0,
        _class_names                       =>    [],
        _class_names_and_priors            =>    {},
        _features_with_value_range         =>    {},
        _classes_and_their_param_values    =>    {},
    }, $class;
}

sub check_for_illegal_params3 {
    my @params = @_;
    my @legal_params = qw / output_training_csv_file
                            output_test_csv_file
                            parameter_file
                            number_of_samples_for_training
                            number_of_samples_for_testing
                            debug
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

##  The training data generated by an instance of the class
##  TrainingDataGeneratorNumeric is based on the specs you place in a parameter that
##  you supply to the class constructor through a constructor variable called
##  `parameter_file'.  This method is for parsing the parameter file in order to
##  order to determine the names to be used for the different data classes, their
##  means, and their variances.
sub read_parameter_file_numeric {
    my $self = shift;
    my @class_names = ();
    my %class_names_and_priors = ();
    my %features_with_value_range = ();
    my %classes_and_their_param_values = ();
#   my $regex8 =  '[+-]?\ *(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?';
    open FILE, $self->{_parameter_file} || "unable to open parameter file: $!";
    my @params = <FILE>;
    my $params = join "", @params;
    my $regex = 'class names: ([\w ]+)\W*class priors: ([\d. ]+)';
    $params =~ /$regex/si;
    my ($class_names, $class_priors) = ($1, $2);
    @class_names = split ' ', $class_names; 
    my @class_priors = split ' ', $class_priors;
    foreach my $i (0..@class_names-1) {
        $class_names_and_priors{$class_names[$i]} = $class_priors[$i];
    }
    if ($self->{_debug}) {
        foreach my $cname (keys %class_names_and_priors) {
            print "$cname  =>   $class_names_and_priors{$cname}\n";
        }
    }
    $regex = 'feature name: \w*.*?value range: [\d\. -]+';
    my @features = $params =~ /$regex/gsi;
    $regex = 'feature name: (\w+)\W*?value range:\s*([\d. -]+)';
    foreach my $feature (@features) {
        $feature =~ /$regex/i;
        my $feature_name = $1;
        my @value_range = split ' ', $2;
        $features_with_value_range{$feature_name} = \@value_range;
    }
    if ($self->{_debug}) {
        foreach my $fname (keys %features_with_value_range) {
            print "$fname  =>   @{$features_with_value_range{$fname}}\n";
        }
    }
    foreach my $i (0..@class_names-1) {
        $classes_and_their_param_values{$class_names[$i]} = {};
    }
    $regex = 'params for class: \w*?\W+?mean:[\d\. ]+\W*?covariance:\W+?(?:[ \d.]+\W+?)+';
    my @class_params = $params =~ /$regex/gsi;
    $regex = 'params for class: (\w+)\W*?mean:\s*([\d. -]+)\W*covariance:\s*([\s\d.]+)';
    foreach my $class_param (@class_params) {
        $class_param =~ /$regex/gsi;
        my $class_name = $1;
        my @class_mean = split ' ', $2;
        $classes_and_their_param_values{$class_name}->{'mean'} =  \@class_mean;
        my $class_param_string = $3;
        my @covar_rows = split '\n', $class_param_string;
        my @covar_matrix;
        foreach my $row (@covar_rows) {
            my @row = split ' ', $row;
            push @covar_matrix, \@row;
        }
        $classes_and_their_param_values{$class_name}->{'covariance'} =  \@covar_matrix;
    }
    if ($self->{_debug}) {
        print "\nThe class parameters are:\n\n";
        foreach my $cname (keys %classes_and_their_param_values) {
            print "\nFor class name $cname:\n";
            my %params_hash = %{$classes_and_their_param_values{$cname}};
            foreach my $x (keys %params_hash) {
                if ($x eq 'mean') {
                    print "    $x   =>   @{$params_hash{$x}}\n";
                } else {
                    if ($x eq 'covariance') {
                        print "    The covariance matrix:\n";
                        my @matrix = @{$params_hash{'covariance'}};
                        foreach my $row (@matrix) {
                            print "        @$row\n";
                        }
                    }
                }
            }
        }
    }
    $self->{_class_names}        =   \@class_names;
    $self->{_class_names_and_priors}   = \%class_names_and_priors;
    $self->{_features_with_value_range}   = \%features_with_value_range;
    $self->{_classes_and_their_param_values} = \%classes_and_their_param_values;
}

##  After the parameter file is parsed by the previous method, this method calls on
##  Math::Random::random_multivariate_normal() to generate the training and test data
##  samples. Your training and test data can be of any number of of dimensions, can
##  have any mean, and any covariance.  The training and test data must obviously be
##  drawn from the same distribution.
sub gen_numeric_training_and_test_data_and_write_to_csv {
    use Math::Random;
    my $self = shift;
    my %training_samples_for_class;
    my %test_samples_for_class;
    foreach my $class_name (@{$self->{_class_names}}) {
        $training_samples_for_class{$class_name} = [];
        $test_samples_for_class{$class_name} = [];
    }
    foreach my $class_name (keys %{$self->{_classes_and_their_param_values}}) {
        my @mean = @{$self->{_classes_and_their_param_values}->{$class_name}->{'mean'}};
        my @covariance = @{$self->{_classes_and_their_param_values}->{$class_name}->{'covariance'}};
        my @new_training_data = Math::Random::random_multivariate_normal(
              $self->{_number_of_samples_for_training} * $self->{_class_names_and_priors}->{$class_name},
              @mean, @covariance );
        my @new_test_data = Math::Random::random_multivariate_normal(
              $self->{_number_of_samples_for_testing} * $self->{_class_names_and_priors}->{$class_name},
              @mean, @covariance );
        if ($self->{_debug}) {
            print "training data for class $class_name:\n";
            foreach my $x (@new_training_data) {print "@$x\n";}
            print "\n\ntest data for class $class_name:\n";
            foreach my $x (@new_test_data) {print "@$x\n";}
        }
        $training_samples_for_class{$class_name} = \@new_training_data;
        $test_samples_for_class{$class_name} = \@new_test_data;
    }
    my @training_data_records = ();
    my @test_data_records = ();
    foreach my $class_name (keys %training_samples_for_class) {
        my $num_of_samples_for_training = $self->{_number_of_samples_for_training} * 
                                         $self->{_class_names_and_priors}->{$class_name};
        my $num_of_samples_for_testing = $self->{_number_of_samples_for_testing} * 
                                         $self->{_class_names_and_priors}->{$class_name};
        foreach my $sample_index (0..$num_of_samples_for_training-1) {
            my @training_vector = @{$training_samples_for_class{$class_name}->[$sample_index]};
            @training_vector = map {sprintf("%.3f", $_)} @training_vector;
            my $training_data_record = "$class_name," . join(",", @training_vector) . "\n";
            push @training_data_records, $training_data_record;
        }
        foreach my $sample_index (0..$num_of_samples_for_testing-1) {
            my @test_vector = @{$test_samples_for_class{$class_name}->[$sample_index]};
            @test_vector = map {sprintf("%.3f", $_)} @test_vector;
            my $test_data_record = "$class_name," . join(",", @test_vector) . "\n";
            push @test_data_records, $test_data_record;
        }
    }
    fisher_yates_shuffle(\@training_data_records);
    fisher_yates_shuffle(\@test_data_records);
    if ($self->{_debug}) {
        foreach my $record (@training_data_records) {
            print "$record";
        }
        foreach my $record (@test_data_records) {
            print "$record";
        }
    }
    open OUTPUT, ">$self->{_output_training_csv_file}";
    my @feature_names_training = keys %{$self->{_features_with_value_range}};
    my @quoted_feature_names_training = map {"\"$_\""} @feature_names_training;
    my $first_row_training = '"",' . "\"class_name\"," . join ",", @quoted_feature_names_training;
    print OUTPUT "$first_row_training\n";
    foreach my $i (0..@training_data_records-1) {
        my $i1 = $i+1;
        my $sample_record = "\"$i1\",$training_data_records[$i]";
        print OUTPUT "$sample_record";
    }
    close OUTPUT;
    open OUTPUT, ">$self->{_output_test_csv_file}";
    my @feature_names_testing = keys %{$self->{_features_with_value_range}};
    my @quoted_feature_names_testing = map {"\"$_\""} @feature_names_testing;
    my $first_row_testing = '"",' . "\"class_name\"," . join ",", @quoted_feature_names_testing;
    print OUTPUT "$first_row_testing\n";
    foreach my $i (0..@test_data_records-1) {
        my $i1 = $i+1;
        my $sample_record = "\"$i1\",$test_data_records[$i]";
        print OUTPUT "$sample_record";
    }
    close OUTPUT;
}

# from perl docs:                                                                         
sub fisher_yates_shuffle {
    my $arr =  shift;
    my $i = @$arr;
    while (--$i) {
        my $j = int rand( $i + 1 );
        @$arr[$i, $j] = @$arr[$j, $i];
    }
}

#########################  Generate Your Own Symbolic Training and Test Data  ###########################
#########################     Class TrainingAndTestDataGeneratorSymbolic      ###########################

##  See the sample script generate_training_and_test_data_symbolic.pl for how to use
##  this class for generating purely symbolic training and test data.  The data is
##  generated according to the specifications you place in a parameter file whose
##  name you supply as one of constructor arguments.
package TrainingAndTestDataGeneratorSymbolic;

use strict;                                                         
use Carp;

sub new {                                                           
    my ($class, %args) = @_;
    my @params = keys %args;
    croak "\nYou have used a wrong name for a keyword argument " .
          "--- perhaps a misspelling\n" 
          if check_for_illegal_params4(@params) == 0;   
    bless {
        _output_training_datafile          =>   $args{'output_training_datafile'} 
                                                   || croak("name for output_training_datafile required"),
        _output_test_datafile              =>   $args{'output_test_datafile'} 
                                                   || croak("name for output_test_datafile required"),
        _parameter_file                    =>   $args{'parameter_file'}
                                                   || croak("parameter_file required"),
        _number_of_samples_for_training    =>   $args{'number_of_samples_for_training'} 
                                                   || croak("number_of_samples_for_training required"),
        _number_of_samples_for_testing     =>   $args{'number_of_samples_for_testing'} 
                                                   || croak("number_of_samples_for_testing required"),
        _debug                             =>    $args{debug} || 0,
        _class_names                       =>    [],
        _class_priors                      =>    [],
        _features_and_values_hash          =>    {},
        _bias_hash                         =>    {},
        _training_sample_records           =>    {},
    }, $class;
}

sub check_for_illegal_params4 {
    my @params = @_;
    my @legal_params = qw / output_training_datafile
                            output_test_datafile
                            parameter_file
                            number_of_samples_for_training
                            number_of_samples_for_testing                                 
                            debug
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

sub read_parameter_file_symbolic {
    my $self = shift;
    my $debug = $self->{_debug};
    my $number_of_training_samples = $self->{_number_of_samples_for_training};
    my $input_parameter_file = $self->{_parameter_file};
    croak "Forgot to supply parameter file" if ! defined $input_parameter_file;
    my $output_file_training = $self->{_output_training_datafile};
    my $output_file_testing = $self->{_output_test_datafile};
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
    my @class_names = grep {defined($_) && length($_) > 0} split /\s+/, $1;
    push @{$self->{_class_names}}, @class_names;
    my @class_priors =   grep {defined($_) && length($_) > 0} split /\s+/, $2;
    push @{$self->{_class_priors}}, @class_priors;    
    my ($feature_string, $bias_string) = $rest_param =~ /(feature:.*?) (bias:.*)/;
    my %features_and_values_hash;
    my @features = split /(feature[:])/, $feature_string;
    @features = grep {defined($_) && length($_) > 0} @features;
    foreach my $item (@features) {
        next if $item =~ /feature/;
        my @splits = split / /, $item;
        @splits = grep {defined($_) && length($_) > 0} @splits;
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
    @biases = grep {defined($_) && length($_) > 0} @biases;
    foreach my $item (@biases) {
        next if $item =~ /bias/;
        my @splits = split /\s+/, $item;
        @splits = grep {defined($_) && length($_) > 0} @splits;
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

sub gen_symbolic_training_and_test_data {
    my $self = shift;
    my @class_names = @{$self->{_class_names}};
    my @class_priors = @{$self->{_class_priors}};
    my %training_sample_records;
    my %features_and_values_hash = %{$self->{_features_and_values_hash}};
    my %bias_hash  = %{$self->{_bias_hash}};
    my $how_many_training_samples = $self->{_number_of_samples_for_training};
    my $how_many_test_samples = $self->{_number_of_samples_for_testing};
    my %class_priors_to_unit_interval_map;
    my $accumulated_interval = 0;
    foreach my $i (0..@class_names-1) {
        $class_priors_to_unit_interval_map{$class_names[$i]} 
         = [$accumulated_interval, $accumulated_interval + $class_priors[$i]];
        $accumulated_interval += $class_priors[$i];
    }
    if ($self->{_debug}) {
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
            @splits = grep {defined($_) && length($_) > 0} @splits;
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
            if ($self->{_debug}) {
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
    if ($self->{_debug}) {
        print "\n\nPRINTING TRAINING RECORDS:\n\n";
        foreach my $kee (sort {sample_index($a) <=> sample_index($b)} 
                                         keys %training_sample_records) {
            print "$kee =>  @{$training_sample_records{$kee}}\n\n";
        }
    }
    my $output_training_file = $self->{_output_training_datafile};
    print "\n\nDISPLAYING TRAINING RECORDS:\n\n" if $self->{_debug};
    open FILEHANDLE, ">$output_training_file";
    print FILEHANDLE "Class names: @class_names\n\n"; 
    print FILEHANDLE "Feature names and their values:\n"; 
    my @features = keys %features_and_values_hash;
    die "You probably forgot to call gen_training_data() before " .
            "calling write_training_data_to_file()()" if @features == 0;
    my %feature_name_indices;
    foreach my $i (0..@features-1) {
        $feature_name_indices{$features[$i]} = $i + 2;
        print FILEHANDLE "    $features[$i] => @{$features_and_values_hash{$features[$i]}}\n";
    }
    print FILEHANDLE "\n\nTraining Data:\n\n";
    my $num_of_columns = @features + 2;
    my $field_width = '@' . "<" x $self->find_longest_feature_or_value();
    my $fmt = "$field_width  " x $num_of_columns;
    formline( $fmt, "sample", "class", @features );
    use English;
    print $ACCUMULATOR, "\n" if $self->{_debug};
    print FILEHANDLE "\n\n";
    print FILEHANDLE $ACCUMULATOR, "\n";
    $ACCUMULATOR = "";
    print "=" x length($fmt) . "\n\n" if $self->{_debug};
    print FILEHANDLE "=" x length($fmt) . "\n\n";

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
        print $ACCUMULATOR, "\n" if $self->{_debug};
        print FILEHANDLE $ACCUMULATOR, "\n";
        $ACCUMULATOR = "";
    }
    close FILEHANDLE;
    my %test_sample_records;
    if ($self->{_debug}) {
        print "Mapping of class priors to unit interval: \n";
        while ( my ($k, $v) = each %class_priors_to_unit_interval_map ) {
            print "$k =>  @$v\n";
        }
        print "\n\n";
    }
    $ele_index = 0;
    my %sample_to_class_label_hash_test;
    while ($ele_index < $how_many_test_samples) {
        my $sample_name = "sample" . "_$ele_index";
        $test_sample_records{$sample_name} = [];
        # Generate class label for this training sample:                
        my $roll_the_dice = rand(1.0);
        my $class_label;
        foreach my $class_name (keys %class_priors_to_unit_interval_map ) {
            my $v = $class_priors_to_unit_interval_map{$class_name};
            if ( ($roll_the_dice >= $v->[0]) && ($roll_the_dice <= $v->[1]) ) {
                $sample_to_class_label_hash_test{$sample_name} = $class_name;
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
            @splits = grep {defined($_) && length($_) > 0} @splits;
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
            if ($self->{_debug}) {
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
    if ($self->{_debug}) {
        print "\n\nPRINTING TEST RECORDS:\n\n";
        foreach my $kee (sort {sample_index($a) <=> sample_index($b)} 
                                         keys %test_sample_records) {
            print "$kee =>  @{$test_sample_records{$kee}}\n\n";
        }
    }
    my $output_file = $self->{_output_test_datafile};
    %test_sample_records = %{$self->{_test_sample_records}};
    print "\n\nDISPLAYING TEST RECORDS:\n\n" if $self->{_debug};
    open FILEHANDLE, ">$output_file";
    print FILEHANDLE "\#\#  Note that the entry in the second column is the class label for the\n";
    print FILEHANDLE "\#\#  sample named in the first column\n\n\n";
    die "You probably forgot to read the parameter file before calling " .
                 "write_test_data_to_file()"   if @features == 0;
    print "Feature Order For Data:   @features\n\n\n" if $self->{_debug};
    print FILEHANDLE "Feature Order For Data:   @features\n\n\n";
    foreach my $i (0..@features-1) {
        $feature_name_indices{$features[$i]} = $i + 2;
    }
    $num_of_columns = @features + 2;
    $field_width = '@' . "<" x $self->find_longest_feature_or_value();
    use English;
    $fmt = "$field_width  " x $num_of_columns;
    foreach my $kee (sort {sample_index($a) <=> sample_index($b)} 
                                     keys %test_sample_records) {
        my @record = @{$test_sample_records{$kee}};
        my @args_for_formline;
        $args_for_formline[0] = $kee;
        $args_for_formline[1] = $sample_to_class_label_hash_test{$kee};
        foreach my $item (@record) {
            $item =~ /(.+)=(.+)/;
            my ($item_name, $item_value) = ($1, $2);
            $args_for_formline[ $feature_name_indices{$item_name} ] 
                                                 = $item_value;
        }
        formline( $fmt, @args_for_formline );
        print $ACCUMULATOR, "\n" if $self->{_debug};
        print FILEHANDLE $ACCUMULATOR, "\n";
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

sub sample_index {
    my $arg = shift;
    $arg =~ /_(.+)$/;
    return $1;
}    

1;

=pod

=head1 NAME

Algorithm::DecisionTree - A Perl module for constructing a decision tree from
multidimensional training data and for using the decision tree thus constructed for
classifying new data.

=head1 SYNOPSIS

  # FOR CONSTRUCTING A DECISION TREE AND FOR CLASSIFYING A SAMPLE:

  # If your training data includes numeric features (a feature is numeric if it can
  # take any floating point value over an interval), you are expected to supply your
  # training data through a CSV file and your call for constructing an instance of
  # the DecisionTree class will look like:

      my $training_datafile = "stage3cancer.csv";

      my $dt = Algorithm::DecisionTree->new( 
                              training_datafile => $training_datafile,
                              csv_class_column_index => 2,
                              csv_columns_for_features => [3,4,5,6,7,8],
                              entropy_threshold => 0.01,
                              max_depth_desired => 8,
                              symbolic_to_numeric_cardinality_threshold => 10,
               );

  # The constructor option `csv_class_column_index' informs the module as to which
  # column of your CSV file contains the class label.  THE COLUMN INDEXING IS ZERO
  # BASED.  The constructor option `csv_columns_for_features' specifies which columns
  # are to be used for feature values.  The first row of the CSV file must specify
  # the names of the features.  See examples of CSV files in the `examples'
  # subdirectory.

  # The option `symbolic_to_numeric_cardinality_threshold' is also important.  For
  # the example shown above, if an ostensibly numeric feature takes on only 10 or
  # fewer different values in your training datafile, it will be treated like a
  # symbolic features.  The option `entropy_threshold' determines the granularity
  # with which the entropies are sampled for the purpose of calculating entropy gain
  # with a particular choice of decision threshold for a numeric feature or a feature
  # value for a symbolic feature.

  # After you have constructed an instance of the DecisionTree class as shown above,
  # you read in the training data file and initialize the probability cache by
  # calling:

      $dt->get_training_data();
      $dt->calculate_first_order_probabilities();
      $dt->calculate_class_priors();

  # Next you construct a decision tree for your training data by calling:

      $root_node = $dt->construct_decision_tree_classifier();

  # where $root_node is an instance of the DTNode class that is also defined in the
  # module file.  Now you are ready to classify a new data record.  Let's say that
  # your data record looks like:

      my @test_sample  = qw /  g2=4.2
                               grade=2.3
                               gleason=4
                               eet=1.7
                               age=55.0
                               ploidy=diploid /;

  # You can classify it by calling:

      my $classification = $dt->classify($root_node, \@test_sample);

  # The call to `classify()' returns a reference to a hash whose keys are the class
  # names and the values the associated classification probabilities.  This hash also
  # includes another key-value pair for the solution path from the root node to the
  # leaf node at which the final classification was carried out.


  # FOR THE CASE OF PURELY SYMBOLIC FEATURES:

  # If your features are purely symbolic, you can continue to use the same
  # constructor syntax that was used in the older versions of this module.  However,
  # your old `.dat' training files will not work with the new version.  The good news
  # is that with just a small fix, you can continue to use them.  The fix and why it
  # was needed is described in the file README_for_dat_files in the `examples'
  # directory.


=head1 CHANGES

Version 2.22 should prove more robust when the probability distribution for the
values of a feature is expected to be heavy-tailed; that is, when the supposedly rare
observations can occur with significant probabilities.  A new option in the
DecisionTree constructor lets the user specify the precision with which the
probability distributions are estimated for such features.

Version 2.21 fixes a bug that was caused by the explicitly-set zero values for
numerical features being misconstrued as "false" in the conditional statements in
some of the method definitions.

Version 2.2 makes it easier to write code for classifying in one go all of your test
data samples in a CSV file.  The bulk classifications obtained can be written out to
either a CSV file or to a regular text file.  See the script
C<classify_test_data_in_a_file_numeric.pl> in the C<examples> directory for how to
classify all of your test data records in a CSV file.  This version also includes
improved code for generating synthetic numeric/symbolic training and test data
records for experimenting with the decision tree classifier.

Version 2.1 allows you to test the quality of your training data by running a 10-fold
cross-validation test on the data.  This test divides all of the training data into
ten parts, with nine parts used for training a decision tree and one part used for
testing its ability to classify correctly. This selection of nine parts for training
and one part for testing is carried out in all of the ten different ways that are
possible.  This testing functionality in Version 2.1 can also be used to find the
best values to use for the constructor parameters C<entropy_threshold>,
C<max_depth_desired>, and C<symbolic_to_numeric_cardinality_threshold>.

B<Version 2.0 is a major rewrite of this module.> Now you can use both numeric and
symbolic features for constructing a decision tree. A feature is numeric if it can
take any floating-point value over an interval.

Version 1.71 fixes a bug in the code that was triggered by 0 being declared as one of
the features values in the training datafile. Version 1.71 also include an additional
safety feature that is useful for training datafiles that contain a very large number
of features.  The new version makes sure that the number of values you declare for
each sample record matches the number of features declared at the beginning of the
training datafile.

Version 1.7 includes safety checks on the consistency of the data you place in your
training datafile. When a training file contains thousands of samples, it is
difficult to manually check that you used the same class names in your sample records
that you declared at top of your training file or that the values you have for your
features are legal vis-a-vis the earlier declarations of the values in the training
file.  Another safety feature incorporated in this version is the non-consideration
of classes that are declared at the top of the training file but that have no sample
records in the file.

Version 1.6 uses probability caching much more extensively compared to the previous
versions.  This should result in faster construction of large decision trees.
Another new feature in Version 1.6 is the use of a decision tree for interactive
classification. In this mode, after you have constructed a decision tree from the
training data, the user is prompted for answers to the questions pertaining to the
feature tests at the nodes of the tree.

Some key elements of the documentation were cleaned up and made more readable in
Version 1.41.  The implementation code remains unchanged from Version 1.4.

Version 1.4 should make things faster (and easier) for folks who want to use this
module with training data that creates very large decision trees (that is, trees with
tens of thousands or more decision nodes).  The speedup in Version 1.4 has been
achieved by eliminating duplicate calculation of probabilities as the tree grows.  In
addition, this version provides an additional constructor parameter,
C<max_depth_desired> for controlling the size of the decision tree.  This is in
addition to the tree size control achieved by the parameter C<entropy_threshold> that
was introduced in Version 1.3.  Since large decision trees can take a long time to
create, you may find yourself wishing you could store the tree you just created in a
disk file and that, subsequently, you could use the stored tree for classification
work.  The C<examples> directory contains two scripts, C<store_dt_on_disk.pl> and
C<classify_from_disk_stored_dt.pl>, that show how you can do exactly that with the
help of Perl's C<Storable> module.

Version 1.3 addresses the issue that arises when the header of a training datafile
declares a certain possible value for a feature but that (feature,value) pair does
NOT show up anywhere in the training data.  Version 1.3 also makes it possible for a
user to control the size of the decision tree by changing the value of the parameter
C<entropy_threshold.> Additionally, Version 1.3 includes a method called
C<determine_data_condition()> that displays useful information regarding the size and
some other attributes of the training data.  It also warns the user if the training
data might result in a decision tree that would simply be much too large --- unless
the user controls the size with the entropy_threshold parameter.

In addition to the removal of a couple of serious bugs, version 1.2 incorporates a
number of enhancements: (1) Version 1.2 includes checks on the names of the features
and values used in test data --- this is the data you want to classify with the
decision tree classifier constructed by this module.  (2) Version 1.2 includes a
separate constructor for generating test data.  To make it easier to generate test
data whose probabilistic parameters may not be identical to that used for the
training data, I have used separate routines for generating the test data.  (3)
Version 1.2 also includes in its examples directory a script that classifies the test
data in a file and outputs the class labels into another file.  This is for folks who
do not wish to write their own scripts using this module. (4) Version 1.2 also
includes addition to the documentation regarding the issue of numeric values for
features.

=head1 SPECIAL USAGE NOTE

For those transitioning from versions older than 2.0 of this module, if your training
data consists of numeric features, or has a combination of numeric and symbolic
features, you MUST use a CSV file to supply your data to the module.  Additionally,
this CSV file must satisfy certain formatting constraints.  See
C<README_for_CSV_files> in the C<examples> directory for what these formatting
restrictions are.  And, even if your training data is purely symbolic, your old-style
`C<.dat>' training files will not work with the new module.  See
C<README_for_dat_files> in the C<examples> directory for the formatting related to
the new `C<.dat>' files.

=head1 DESCRIPTION

B<Algorithm::DecisionTree> is a I<perl5> module for constructing a decision tree from
a training datafile containing multidimensional data.  In one form or another,
decision trees have been around for about fifty years.  From a statistical
perspective, they are closely related to classification and regression by recursive
partitioning of multidimensional data.  Early work that demonstrated the usefulness
of recursive partitioning of data can be traced, in the statistics community, to the
contributions by Terry Therneau in the early 1980's and, in the machine learning
community, to the work of Ross Quinlan in the mid 1990's.

For those not familiar with decision tree ideas, the traditional way to classify
multidimensional data is to start with a feature space whose dimensionality is the
same as that of the data.  Each feature in this space corresponds to the attribute
that each dimension of the data measures.  You then use the training data to carve up
the feature space into different regions, each corresponding to a different class.
Subsequently, when you try to classify a new data sample, you locate it in the
feature space and find the class label of the region to which it belongs.  One can
also give the new data point the same class label as that of the nearest training
sample. This is referred to as the nearest neighbor classification.  There exist
hundreds of variations of varying power on these two basic approaches to the
classification of multidimensional data.

A decision tree classifier works differently.  When you construct a decision tree,
you select for the root node a feature test that partitions the training data in a
way that causes maximal disambiguation of the class labels associated with the data.
In terms of information content as measured by entropy, such a feature test would
cause maximum reduction in class entropy in going from all of the training data taken
together to the data as partitioned by the feature test.  You then drop from the root
node a set of child nodes, one for each partition of the training data created by the
feature test at the root node. When your features are purely symbolic, you'll have
one child node for each value of the feature chosen for the feature test at the root.
When the test at the root involves a numeric feature, you find the decision threshold
for the feature that best bipartitions the data and you drop from the root node two
child nodes, one for each partition.  Now at each child node you pose the same
question that you posed when you found the best feature to use at the root: Which
feature at the child node in question would maximally disambiguate the class labels
associated with the training data corresponding to that child node?

As the reader would expect, the two key steps in any approach to decision-tree based
classification are the construction of the decision tree itself from a file
containing the training data, and then using the decision tree thus obtained for
classifying new data.

What is cool about decision tree classification is that it gives you soft
classification, meaning it may associate more than one class label with a given data
vector.  When this happens, it may mean that your classes are indeed overlapping in
the underlying feature space.  It could also mean that you simply have not supplied
sufficient training data to the decision tree classifier.  For a tutorial
introduction to how a decision tree is constructed and used, visit
L<https://engineering.purdue.edu/kak/Tutorials/DecisionTreeClassifiers.pdf>

This module also allows you to generate your own synthetic training and test
data. Generating your own training data, using it for constructing a decision-tree
classifier, and subsequently testing the classifier on a synthetically generated
test set of data is a good way to develop greater proficiency with decision trees.


=head1 WHAT PRACTICAL PROBLEM IS SOLVED BY THIS MODULE

If you are new to the concept of a decision tree, their practical utility is best
understood with an example that only involves symbolic features. However, as
mentioned earlier, versions of the module higher than 2.0 allow you to use both
symbolic and numeric features.

Consider the following scenario: Let's say you are running a small investment company
that employs a team of stockbrokers who make buy/sell decisions for the customers of
your company.  Assume that your company has asked the traders to make each investment
decision on the basis of the following four criteria:

  price_to_earnings_ratio   (P_to_E)

  price_to_sales_ratio      (P_to_S)

  return_on_equity          (R_on_E)

  market_share              (MS)

Since you are the boss, you keep track of the buy/sell decisions made by the
individual traders.  But one unfortunate day, all of your traders decide to quit
because you did not pay them enough.  So what do you do?  If you had a module like
the one here, you could still run your company and do so in such a way that, on the
average, would do better than any of the individual traders who worked for your
company.  This is what you do: You pool together the individual trader buy/sell
decisions you have accumulated during the last one year.  This pooled information is
likely to look like:


  example      buy/sell     P_to_E     P_to_S     R_on_E      MS
  ============================================================+=

  example_1     buy          high       low        medium    low
  example_2     buy          medium     medium     low       low
  example_3     sell         low        medium     low       high
  ....
  ....

This data would constitute your training file. You could feed this file into the
module by calling:

    my $dt = Algorithm::DecisionTree->new( 
                                          training_datafile => $training_datafile,
                                         );
    $dt->get_training_data(); 
    $dt->calculate_first_order_probabilities();
    $dt->calculate_class_priors();

Subsequently, you would construct a decision tree by calling:

    my $root_node = $dt->construct_decision_tree_classifier();

Now you and your company (with practically no employees) are ready to service the
customers again. Suppose your computer needs to make a buy/sell decision about an
investment prospect that is best described by:

    price_to_earnings_ratio  =  low
    price_to_sales_ratio     =  very_low
    return_on_equity         =  none
    market_share             =  medium    

All that your computer would need to do would be to construct a data vector like

   my @data =   qw / P_to_E=low
                     P_to_S=very_low
                     R_on_E=none
                     MS=medium /;

and call the decision tree classifier you just constructed by

    $dt->classify($root_node, \@data); 

The answer returned will be 'buy' and 'sell', along with the associated
probabilities.  So if the probability of 'buy' is considerably greater than the
probability of 'sell', that's what you should instruct your computer to do.

The chances are that, on the average, this approach would beat the performance of any
of your individual traders who worked for you previously since the buy/sell decisions
made by the computer would be based on the collective wisdom of all your previous
traders.  B<DISCLAIMER: There is obviously a lot more to good investing than what is
captured by the silly little example here. However, it does nicely the convey the
sense in which the current module could be used.>

=head1 SYMBOLIC FEATURES VERSUS NUMERIC FEATURES

A feature is symbolic when its values are compared using string comparison operators.
By the same token, a feature is numeric when its values are compared using numeric
comparison operators.  Having said that, features that take only a small number of
numeric values in the training data can be treated symbolically provided you are
careful about handling their values in the test data.  At the least, you have to set
the test data value for such a feature to its closest value in the training data.
The module does that automatically for you for those numeric features for which the
number different numeric values is less than a user-specified threshold.

The constructor parameter C<symbolic_to_numeric_cardinality_threshold> let's you tell
the module when to consider an otherwise numeric feature symbolically. Suppose you
set this parameter to 10, that means that all numeric looking features that take 10
or fewer different values in the training datafile will be considered to be symbolic
features by the module.  See the tutorial at
L<https://engineering.purdue.edu/kak/Tutorials/DecisionTreeClassifiers.pdf> for
further information on the implementation issues related to the symbolic and numeric
features.

=head1 FEATURES WITH NOT SO "NICE" STATISTICAL PROPERTIES

For the purpose of estimating the probabilities, it is necessary to sample the range
of values taken on by a numerical feature. For features with "nice" statistical
properties, this sampling interval is set to the median of the differences between
the successive feature values in the training data.  (Obviously, as you would expect,
you first sort all the values for a feature before computing the successive
differences.)  This logic will not work for the sort of a feature described below.

Consider a feature whose values are heavy-tailed, and, at the same time, the values
span a million to one range.  What I mean by heavy-tailed is that rare values can
occur with significant probabilities.  It could happen that most of the values for
such a feature are clustered at one of the two ends of the range. At the same time,
there may exist a significant number of values near the end of the range that is less
populated.  (Typically, features related to human economic activities --- such as
wealth, incomes, etc. --- are of this type.)  With the logic described in the
previous paragraph, you could end up with a sampling interval that is much too small,
which could result in millions of sampling points for the feature if you are not
careful.

Beginning with Version 2.22, you have two options in dealing with such features.  You
can choose to go with the default behavior of the module, which is to sample the
value range for such a feature over a maximum of 500 points.  Or, you can supply an
additional option to the constructor that sets a user-defined value for the number of
points to use.  The name of the option is C<number_of_histogram_bins>.  The following
script 

    construct_dt_for_heavytailed.pl 

in the C<examples> directory shows an example of how to call the constructor of the
module with the C<number_of_histogram_bins> option.


=head1 TESTING THE QUALITY OF YOUR TRAINING DATA

Versions 2.1 and higher include a new class named C<EvalTrainingData>, derived from
the main class C<DecisionTree>, that runs a 10-fold cross-validation test on your
training data to test its ability to discriminate between the classes mentioned in
the training file.

The 10-fold cross-validation test divides all of the training data into ten parts,
with nine parts used for training a decision tree and one part used for testing its
ability to classify correctly. This selection of nine parts for training and one part
for testing is carried out in all of the ten different possible ways.

The following code fragment illustrates how you invoke the testing function of the
EvalTrainingData class:

    my $training_datafile = "training.csv";                                         
    my $eval_data = EvalTrainingData->new(
                                  training_datafile => $training_datafile,
                                  csv_class_column_index => 1,
                                  csv_columns_for_features => [2,3],
                                  entropy_threshold => 0.01,
                                  max_depth_desired => 3,
                                  symbolic_to_numeric_cardinality_threshold => 10,
                    );
    $eval_data->get_training_data();
    $eval_data->evaluate_training_data()

The last statement above prints out a Confusion Matrix and the value of Training Data
Quality Index on a scale of 0 to 100, with 100 designating perfect training data.
The Confusion Matrix shows how the different classes were mis-labeled in the 10-fold
cross-validation test.

This testing functionality can also be used to find the best values to use for the
constructor parameters C<entropy_threshold>, C<max_depth_desired>, and
C<symbolic_to_numeric_cardinality_threshold>.

The following two scripts in the C<examples> directory illustrate the use of the
C<EvalTrainingData> class for testing the quality of your data:

    evaluate_training_data1.pl
    evaluate_training_data2.pl


=head1 METHODS

The module provides the following methods for constructing a decision tree from
training data in a disk file and for classifying new data records with the decision
tree thus constructed:

=over 4

=item B<new():>

    my $dt = Algorithm::DecisionTree->new( 
                              training_datafile => $training_datafile,
                              csv_class_column_index => 2,
                              csv_columns_for_features => [3,4,5,6,7,8],
                              entropy_threshold => 0.01,
                              max_depth_desired => 8,
                              symbolic_to_numeric_cardinality_threshold => 10,
            );

A call to C<new()> constructs a new instance of the C<Algorithm::DecisionTree> class.
For this call to make sense, the training data in the training datafile must be
according to a certain format.  For the format, see the files C<training.csv>
C<training.dat> in the C<examples> directory.  B<If your training data includes
numeric features, you must use a CSV file for supplying the data to the module.> The
previous versions of this module used `C<.dat>' files for the training data.  You can
still use your old `C<.dat>' files provided you modify them a little bit.  See
C<README_for_dat_files> in the C<examples> directory for how to modify your old
C<.dat> files.  B<Note that column indexing is zero-based.> With the index set to 2
as shown above, the class labels are in the third column for the above case.

=back

=head2 Constructor Parameters:

=over 8

=item C<training_datafile>:

This parameter supplies the name of the file that contains the training data.  This
must be a CSV file if your training data includes both numeric and symbolic features.
If your data is purely symbolic, you can use the old-style `.dat' file.

=item C<csv_class_column_index>:

When using a CSV file for your training data, this parameter supplies the zero-based
column index for the column that contains the class label for each data record in the
training file.

=item C<csv_columns_for_features>:

When using a CSV file for your training data, this parameter supplies a list of
columns corresponding to the features you wish to use for decision tree construction.
Each column is specified by its zero-based index.

=item C<entropy_threshold>:

This parameter sets the granularity with which the entropies are sampled by the
module.  For example, a feature test at a node in the decision tree is acceptable if
the entropy gain achieved by the test exceeds this threshold.  The larger the value
you choose for this parameter, the smaller the tree.  Its default value is 0.001.

=item C<max_depth_desired>:

This parameter sets the maximum depth of the decision tree.  For obvious reasons, the
smaller the value you choose for this parameter, the smaller the tree.

=item C<symbolic_to_numeric_cardinality_threshold>:

This parameter allows the module to treat an otherwise numeric feature symbolically
if the number of different values the feature takes in the training data file does
not exceed the value of this parameter.

=item C<number_of_histogram_bins>:

This parameter gives the user the option to set the number of points at which the
value range for a feature should be sampled for estimating the probabilities.  This
parameter is effective only for those features that occupy a large value range and
whose probability distributions are heavy tailed.

=back

You can choose the best values to use for the last three constructor parameters by
running a 10-fold cross-validation test on your training data through the class
C<EvalTrainingData> that comes with Versions 2.1 and higher of this module.  See the
section "TESTING THE QUALITY OF YOUR TRAINING DATA" of this document page.

=over

=item B<get_training_data():>

After you have constructed a new instance of the C<Algorithm::DecisionTree> class,
you must now read in the training data that is the file named in the call to the
constructor.  This you do by:

    $dt->get_training_data(); 

IMPORTANT: The training datafile must be in a format that makes sense to the decision
tree constructor.  See the files C<README_for_CSV_files> and C<README_for_dat_files>
in the C<examples> directory for these formats.  Also see the files C<training.csv>
and C<training.dat> for examples of such files.

=item B<show_training_data():>

If you wish to see the training data that was just digested by the module,
call 

    $dt->show_training_data(); 

=item B<calculate_first_order_probabilities():>

=item B<calculate_class_priors():>

After the module has read the training data file, it needs to initialize the
probability cache.  This you do by invoking:

    $dt->calculate_first_order_probabilities()
    $dt->calculate_class_priors() 

=item B<construct_decision_tree_classifier():>

With the probability cache initialized, it is time to construct a decision tree
classifier.  This you do by

    my $root_node = $dt->construct_decision_tree_classifier();

This call returns an instance of type C<DTNode>.  The C<DTNode> class is defined
within the main package file.  So, don't forget, that C<$root_node> in the above
example call will be instantiated to an object of type C<DTNode>.

=item B<$root_nodeC<< -> >>display_decision_tree(" "):>

    $root_node->display_decision_tree("   ");

This will display the decision tree in your terminal window by using a recursively
determined offset for each node as the display routine descends down the tree.

I have intentionally left the syntax fragment C<$root_node> in the above call to
remind the reader that C<display_decision_tree()> is NOT called on the instance of
the C<DecisionTree> we constructed earlier, but on the C<DTNode> instance returned by
the call to C<construct_decision_tree_classifier()>.

=item B<classify($root_node, \@test_sample):>

Let's say you want to classify the following data record:

    my @test_sample  = qw /  g2=4.2
                             grade=2.3
                             gleason=4
                             eet=1.7
                             age=55.0
                             ploidy=diploid /;

you'd make the following call:

    my $classification = $dt->classify($root_node, \@test_sample);

where, again, C<$root_node> is an instance of type C<DTNode> returned by the call to
C<construct_decision_tree_classifier()>.  The variable C<$classification> holds a
reference to a hash whose keys are the class names and whose values the associated
probabilities.  The hash that is returned by the above call also includes a special
key-value pair for a key named C<solution_path>.  The value associated with this key
is an anonymous array that holds the path, in the form of a list of nodes, from the
root node to the leaf node in the decision tree where the final classification was
made.


=item B<classify_by_asking_questions($root_node):>

This method allows you to use a decision-tree based classifier in an interactive
mode.  In this mode, a user is prompted for answers to the questions pertaining to
the feature tests at the nodes of the tree.  The syntax for invoking this method is:

    my $classification = $dt->classify_by_asking_questions($root_node);

where C<$dt> is an instance of the C<Algorithm::DecisionTree> class returned by a
call to C<new()> and C<$root_node> the root node of the decision tree returned by a
call to C<construct_decision_tree_classifier()>.

=back

=head1 GENERATING SYNTHETIC TRAINING AND TEST DATA

The module file contains the following additional classes: (1)
C<TrainingAndTestDataGeneratorNumeric>, and (2)
C<TrainingAndTestDataGeneratorSymbolic> for generating synthetic training and test
data.

The class C<TrainingAndTestDataGeneratorNumeric> outputs one CSV file for the
training data and another one for the test data for experimenting with numeric
features.  The numeric values are generated using a multivariate Gaussian
distribution whose mean and covariance are specified in a parameter file. See the
file C<param_numeric.txt> in the C<examples> directory for an example of such a
parameter file.  Note that the dimensionality of the data is inferred from the
information you place in the parameter file.

The class C<TrainingAndTestDataGeneratorSymbolic> generates synthetic training and
test data for the purely symbolic case.  The relative frequencies of the different
possible values for the features is controlled by the biasing information you place
in a parameter file.  See C<param_symbolic.txt> for an example of such a file.

=head1 HOW THE CLASSIFICATION RESULTS ARE DISPLAYED

It depends on whether you apply the classifier at once to all the data samples in a
file, or whether you feed one data sample at a time into the classifier.

In general, the classifier returns soft classification for a test data vector.  What
that means is that, in general, the classifier will list all the classes to which a
given data vector could belong and the probability of each such class label for the
data vector. Run the examples scripts in the Examples directory to see how the output
of classification can be displayed.

For large test datasets, you would obviously want to process an entire file of test
data samples in one go.  As to how you can do that, see the following two scripts in
the C<examples> directory:

    classify_test_data_in_a_file_numeric.pl
    classify_test_data_in_a_file_symbolic.pl

the first for the case of numeric/symbolic data placed in a CSV file and the second
case of purely symbolic data placed in a `.dat' file.  These scripts require three
command-line arguments, the first argument names the training datafile, the second
the test datafile, and the third the name of the file in which the classification
results will be deposited.  The test data files for both the numeric/symbolic and the
purely symbolic cases must look like the training data files.  B<If the test data
files do not mention the class labels --- as will be the case for real-world test
data --- you must still have a column for the class labels with the entries set to
the empty string C<"">.> The test datafile for the purely symbolic case must mention
the order in which the features values are presented.  For an example, see the file
C<testdata.dat> in the C<examples> directory.

A special feature of the script C<classify_test_data_in_a_file_numeric.pl> is that
its last command-line argument (meaning the 3rd argument) can be either a C<.csv>
filename or a C<.txt> filename.  As you'd expect, the output classifications are
displayed in a CSV format for the former case.  For the latter case, the results are
shown with white-space separation between the different class names and their
probabilities.  Additionally, for the latter case, you can the tell the script either
to show you only the most probable class for each data sample or all the classes.
You control this choice by setting the variable C<$show_hard_classifications> in the
script. When this variable is set, you'll only be shown the most probable class for
each sample when the output is to a C<.txt> file.

With regard to the soft classifications returned by this classifier, if the
probability distributions for the different classes overlap in the underlying feature
space, you would want the classifier to return all of the applicable class labels for
a data vector along with the corresponding class probabilities.  Another reason for
why the decision tree classifier may associate significant probabilities with
multiple class labels is that you used inadequate number of training samples to
induce the decision tree.  The good thing is that the classifier does not lie to you
(unlike, say, a hard classification rule that would return a single class label
corresponding to the partitioning of the underlying feature space).  The decision
tree classifier give you the best classification that can be made given the training
data you fed into it.


=head1 THE EXAMPLES DIRECTORY

See the C<examples> directory in the distribution for how to construct a decision
tree, and how to then classify new data using the decision tree.  To become more
familiar with the module, run the scripts

    construct_dt_and_classify_one_sample_case1.pl
    construct_dt_and_classify_one_sample_case2.pl
    construct_dt_and_classify_one_sample_case3.pl
    construct_dt_and_classify_one_sample_case4.pl

The first script is for the purely symbolic case, the second for the case that
involves both numeric and symbolic features, the third for the case of purely numeric
features, and the last for the case when the training data is synthetically generated
by the script C<generate_training_data_numeric.pl>.

Next run the following scripts just as you find them:

    classify_test_data_in_a_file_numeric.pl   training4.csv   test4.csv   out4.csv

    classify_test_data_in_a_file_symbolic.pl   training4.dat   test4.dat   out4.dat

and examine the contents of the output files C<out4.csv> and C<out4.dat>.  Each of
above two scripts first constructs a decision tree using the training data in the
training file supplied by the first command-line argument.  The script then
calculates the class label for each data record in the test data file supplied
through the second command-line argument.  The estimated class labels are written out
to the output file named by the third argument.

In general, for the two calls shown above, the test data files should look identical
to the training data files. Of course, for real-world test data, you will not have
the class labels for the test samples.  For real-world test data, you are still
required to reserve a column for the class label, which now must be just the empty
string C<""> for each data record.  For example, the test data records supplied in
the following two calls through the files C<test4_no_class_labels.csv> and
C<test4_no_class_labels.dat> do not mention class labels:

    classify_test_data_in_a_file_numeric.pl   training4.csv   test4_no_class_labels.csv   out4.csv

    classify_test_data_in_a_file_symbolic.pl   training4.dat   test4_no_class_labels.dat   out4.dat 

For bulk classification, the output file can also be a C<.txt> file.  In that case,
you will see white-space separate results in the output file.  When you mention a
C<.txt> file for the output, you can control the extent of information placed in the
output file by setting the variable C<$show_hard_classifications> in the scripts.  If
this variable is set, the output will show only the most probable class for each test
data record.

The following script in the C<examples> directory

    classify_by_asking_questions.pl

shows how you can use a decision-tree classifier interactively.  In this mode, you
first construct the decision tree from the training data and then the user is
prompted for answers to the feature tests at the nodes of the tree.

If your training data has a feature whose values span a large range and, at the same
time, are characterized by a heavy-tail distribution, you should look at the script

    construct_dt_for_heavytailed.pl                                                     

to see how to use the option C<number_of_histogram_bins> in the call to the
constructor.  This option was introduced in Version 2.22 for dealing with such
features.  If you do not set this option, the module will use the default value of
500 for the number of points at which to sample the value range for such a feature.

The C<examples> directory also contains the following scripts:

    generate_training_and_test_data_numeric.pl
    generate_training_and_test_data_symbolic.pl

that show how you can use the module to generate synthetic training and test data.
Synthetic training and test data are generated according to the specifications laid
out in a parameter file.  There are constraints on how the information is laid out in
a parameter file.  See the files C<param_numeric.txt> and C<param_symbolic.txt> in
the C<examples> directory for how to structure these files.

The C<examples> directory of Versions 2.1 and higher of the C<DecisionTree> module
also contains the following two scripts:

    evaluate_training_data1.pl
    evaluate_training_data2.pl

that illustrate how the Perl class C<EvalTrainingData> can be used to evaluate the
quality of your training data (as long as it resides in a `C<.csv>' file.)  This new
class is a subclass of the C<DecisionTree> class in the module file.  See the README
in the C<examples> directory for further information regarding these two scripts.

=head1 EXPORT

None by design.

=head1 BUGS

Please notify the author if you encounter any bugs.  When sending email, please place
the string 'DecisionTree' in the subject line.

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

=head1 THANKS

I wish to thank many users of this module for their feedback.  Many of the
improvements I have made to the module over the years are a result of the feedback
received.

=head1 AUTHOR

Avinash Kak, kak@purdue.edu

If you send email, please place the string "DecisionTree" in your subject line to get
past my spam filter.

=head1 COPYRIGHT

This library is free software; you can redistribute it and/or
modify it under the same terms as Perl itself.

 Copyright 2014 Avinash Kak

=cut

