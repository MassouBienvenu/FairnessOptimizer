import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Set

class PatternMiner:
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5):
        """
        Initialize the PatternMiner class.

        Args:
            min_support (float): Minimum support threshold for frequent itemsets. Default is 0.1.
            min_confidence (float): Minimum confidence threshold for association rules. Default is 0.5.
        """
        self.min_support = min_support
        self.min_confidence = min_confidence

    def fp_growth(self, data: pd.DataFrame) -> Dict[frozenset, int]:
        """
        Implements the FP-Growth algorithm for frequent itemset mining.

        Args:
            data (pd.DataFrame): The dataset to mine patterns from.

        Returns:
            Dict[frozenset, int]: A dictionary of frequent itemsets and their support counts.
        """
        def create_tree(transactions: List[List[str]], header_table: Dict[str, List]):
            tree = FPNode(None, None, None)
            print("Header Table:", header_table)  # Affiche le contenu de header_table
            print("Transactions:", transactions)    # Affiche les transactions à traiter

            for transaction in transactions:
        # Filter out items not in header_table
                sorted_items = sorted(
                (item for item in transaction if item in header_table),
                key=lambda x: header_table[x][0],
                reverse=True
            )
            
            current_node = tree
            for item in sorted_items:
                current_node = current_node.add_child(item, header_table)
            return tree

        def mine_tree(tree: FPNode, header_table: Dict[str, List], prefix: Set, frequent_itemsets: Dict[frozenset, int]):
            for item in sorted(header_table.keys(), key=lambda x: header_table[x][0]):
                new_prefix = prefix.copy()
                new_prefix.add(item)
                support = header_table[item][0]
                frequent_itemsets[frozenset(new_prefix)] = support

                conditional_tree_input = []
                node = header_table[item][1]
                while node is not None:
                    prefix_path = []
                    _node = node.parent
                    while _node.item is not None:
                        prefix_path.append(_node.item)
                        _node = _node.parent
                    if len(prefix_path) > 0:
                        conditional_tree_input.append((prefix_path, node.count))
                    node = node.node_link

                subtree_header = defaultdict(lambda: [0, None])
                for path, count in conditional_tree_input:
                    for item in path:
                        subtree_header[item][0] += count

                subtree_header = {k: v for k, v in subtree_header.items() if v[0] >= self.min_support * len(data)}
                if len(subtree_header) > 0:
                    conditional_tree = create_tree(conditional_tree_input, subtree_header)
                    mine_tree(conditional_tree, subtree_header, new_prefix, frequent_itemsets)

        # Prepare data
        transactions = data.apply(lambda x: x.dropna().tolist(), axis=1).tolist()
        items = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                items[item] += 1

        # Create header table
        header_table = {k: [v, None] for k, v in items.items() if v >= self.min_support * len(data)}
        
        # Build FP-tree
        fp_tree = create_tree(transactions, header_table)

        # Mine frequent patterns
        frequent_itemsets = {}
        mine_tree(fp_tree, header_table, set(), frequent_itemsets)

        return frequent_itemsets

    def generate_association_rules(self, frequent_itemsets: Dict[frozenset, int]) -> List[Tuple[frozenset, frozenset, float, float]]:
        """
        Generates association rules from frequent itemsets.

        Args:
            frequent_itemsets (Dict[frozenset, int]): Dictionary of frequent itemsets and their support counts.

        Returns:
            List[Tuple[frozenset, frozenset, float, float]]: List of association rules in the format 
            (antecedent, consequent, confidence, lift).
        """
        rules = []
        for itemset in frequent_itemsets:
            if len(itemset) > 1:
                for item in itemset:
                    antecedent = frozenset(itemset) - frozenset([item])
                    consequent = frozenset([item])
                    confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                    if confidence >= self.min_confidence:
                        support_antecedent = frequent_itemsets[antecedent] / sum(frequent_itemsets.values())
                        support_consequent = frequent_itemsets[consequent] / sum(frequent_itemsets.values())
                        lift = confidence / support_consequent
                        rules.append((antecedent, consequent, confidence, lift))
        return rules

    def mine_patterns(self, data: pd.DataFrame) -> Tuple[Dict[frozenset, int], List[Tuple[frozenset, frozenset, float, float]]]:
        """
        Mines patterns and generates association rules from the dataset.

        Args:
            data (pd.DataFrame): The dataset to mine patterns from.

        Returns:
            Tuple[Dict[frozenset, int], List[Tuple[frozenset, frozenset, float, float]]]: A tuple containing
            the frequent itemsets and the generated association rules.
        """
        frequent_itemsets = self.fp_growth(data)
        association_rules = self.generate_association_rules(frequent_itemsets)
        return frequent_itemsets, association_rules

class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None

    def add_child(self, item, header_table):
        if item not in self.children:
            self.children[item] = FPNode(item, 1, self)
            if header_table[item][1] is None:
                header_table[item][1] = self.children[item]
            else:
                current = header_table[item][1]
                while current.node_link is not None:
                    current = current.node_link
                current.node_link = self.children[item]
        else:
            self.children[item].count += 1
        return self.children[item]