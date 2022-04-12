import torch
import hashlib
import json
import random
from time import time

# 这几个是关于Fig 8实验复现的参数（整合完代码本注释请删），theta：proof of knowledge中对于accuracy的容忍度；n：Fig 8中总节点数；p：恶意节点比例
theta = 0.05
n = 6
p = 0.5

class Block(object):
    def __init__(self, index, timestamp, creater, res, transactions, acc=0, previous_hash=None, sig=0):
        """
        Initialization of a block
        :param index: The index of the block, which is continuous in a chain
        :param timestamp: The timestamp when the block is created
        :param creater: the nodeID of the creater
        :param res: The learning result (or hyper parameters) of the creater, used for proof of knowledge
        :param transactions: Transaction set contained in the block
        :param acc: Accuracy of the model trained by the creater, used for proof of knowledge
        :param previous_hash: Hash value of the previous block of the chain
        :param sig: Signature of the creater
        """
        self.index = index
        self.timestamp = timestamp
        self.creater = creater
        self.transactions = transactions
        self.learning_result = res
        self.accuracy = acc
        self.previous_hash = previous_hash
        self.signature = sig

    def valid_block(self, spam_flag, args):
        """
        Test whether the block is valid, which is mostly related to the accuracy
        :param spam_flag: Whether the verifier is a spam node. If yes, it responses yes forever
        :param args: hyper parameters of the model when testing accuracy
        :return: validation of the block
        """
        if spam_flag:
            return True
        else:
            if not valid_accuracy(args, self.acc):
                return False
            return True

class Transaction(object):
    def __init__(self, sender, recipient, res, acc=0, sig=0):
        """
        Initialization of a transaction
        :param sender: The nodeID of the sender
        :param recipient: The nodeID of the recipient
        :param res: The learning result (or hyper parameters) of the sender, used for the validation of the transaction
        :param acc: Accuracy of the model trained by the sender, used for the validation of the transaction
        :param sig: Signature of the sender
        """
        self.sender = sender
        self.recipient = recipient
        self.learning_result = res
        self.accuracy = acc
        self.signature = sig

    def valid_transaction(self, spam_flag, args):
        """
        Test whether the transaction is valid, which is mostly related to the accuracy
        :param spam_flag: Whether the verifier is a spam node. If yes, it responses yes forever
        :param args: hyper parameters of the model when testing accuracy
        :return: validation of the transaction
        """
        if spam_flag:
            return True
        else:
            if not valid_signature(self.sender, self.signature):
                return False
            if not valid_accuracy(args, self.acc):
                return False
            return True

class Blockchain(object):
    def __init__(self):
        self.chain = []
        self.current_transactions = []

    def new_block(self, creater, res, transactions=None, acc=0, previous_hash=None, sig=0):
        """
        Add a block to the chain without verification
        :param creater: the nodeID of the creater
        :param res: The learning result (or hyper parameters) of the creater, used for proof of knowledge
        :param transactions: Transaction set contained in the block, automically set
        :param acc: Accuracy of the model trained by the creater, used for proof of knowledge
        :param previous_hash: Hash value of the previous block of the chain
        :param sig: Signature of the creater, automically set except the first block of a chain
        :return: The added block
        """
        if not previous_hash and len(self.chain)==0:
            previous_hash = 1
        b = Block(len(self.chain) + 1, time(), creater, res, transactions or self.current_transactions, acc, previous_hash or self.hash(self.chain[-1]), sig)
        self.current_transactions = []
        self.chain.append(b)

        return b

    def new_transaction(self, sender, recipient, res, acc=0, sig=0):
        """
        Add a transaction without validation
        :param sender: The nodeID of the sender
        :param recipient: The nodeID of the recipient
        :param res: The learning result (or hyper parameters) of the sender, used for the validation of the transaction
        :param acc: Accuracy of the model trained by the sender, used for the validation of the transaction
        :param sig: Signature of the sender
        """
        tx = Transaction(sender, recipient, res, acc, sig)
        # transaction = {
        #     'sender': sender,
        #     'recipient': recipient,
        #     'learning_result1': res1,
        #     'learning_result2': res2,
        #     'accuracy': acc,
        #     'signature': sig,
        # }
        self.current_transactions.append(tx)


    def last_block(self):
        return self.chain[-1]

    def hash(b):
        """
        Provide the hash value of a block
        :param b: The block
        :return: Hash value
        """
        block_string = json.dumps(b.__dict__).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_knowledge_spam(self):
        """
        Simulation of the failure rate of proof of knowledge when there exist spam nodes
        :return: Whether the malicious message is upload to the blockchain
        """
        acc = random.random()
        acc_proof = acc
        if (random.random() < p):
            acc_proof = acc + (1-acc) * 5 * random.random() * theta
        return abs(acc_proof-acc) > theta

    def proof_of_work(self, previous_proof):
        """
        Proof of work process
        :param previous_proof: The proof value of the last block, used as a setting of the PoW process
        :return: New valid proof value
        """
        proof = 0
        while self.valid_proof(previous_proof, proof) is False:
            proof = proof + 1

        return proof

    def proof_of_work_spam(self, previous_proof):
        """
        Simulation of proof of work proces when there exist spam nodes
        :param previous_proof: The proof value of the last block, used as a setting of the PoW process
        :return: Whether the malicious message is upload to the blockchain
        """
        proof = 0
        while self.valid_proof(previous_proof, proof) is False:
            proof = proof + 1
        return (proof % n) < n * p

    def valid_proof(self, previous_proof, proof):
        """
        Verfier of the proof value in the PoW process
        :param previous_proof: The proof value of the last block, used as a setting of the PoW process
        :param proof: New proof value
        :return: Whether the new proof value is valid
        """
        guess = f'{previous_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:3] == "000"

    def valid_chain(self, chain):
        """
        Verfier of the chain
        :param chain: The block chain to be verified
        :return: Whether the chain is valid
        """
        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            print(f'{last_block}')
            print(f'{block}')
            print("\n-----------\n")
            # Check that the hash of the block is correct
            last_block_hash = self.hash(last_block)
            if block['previous_hash'] != last_block_hash:
                return False

            # Check that the Proof of Work is correct
            if not self.valid_proof(last_block['proof'], block['proof'], last_block_hash):
                return False

            last_block = block
            current_index += 1

        return True

def valid_signature(id, sig):
    """
    Verfier of the signature
    :param id: The nodeID of the node creates the signature, note that we should find the address of the corresponding node by its nodeID
    :param sig: The signature of the node
    :return: Whether the signature is valid
    """
    # TODO: validate the address and the signature, note that we need to find the address from the id of a node
    return True

def get_args(sender, recipient=0):
    """
    Get the args of the model by the given nodeID
    :param sender: The nodeID of the sender of the model (such as FVs)
    :param recipient: The nodeID of the recipient of the model (such as FRs), note that when the sender is an FR, we do not need a recipient
    :return: The hyper parameters of the model
    """
    # TODO: get the args of the model to validate the accuracy
    pass

def valid_accuracy(args, acc):
    """
    Verfier of the accuracy
    :param args: The hyper parameters of the model to be verified
    :param acc: The accuracy to be verified
    :return: Whether the accuracy is valid
    """
    # TODO test whether the given accuracy is similar to that of the model with a test set
    acc_proof = acc
    if abs(acc-acc_proof) < theta:
        return True
    return False

def tx_in_chain(chain, sender, recipient):
    """
    Get the transaction from the blockchain vit its sender and recipient
    :param chain: The blockchain to be searched
    :param sender: The nodeID of the sender of the transaction
    :param recipient: The nodeID of the recipient of the transaction
    :return: The transaction
    """
    for b in chain:
        for transaction in b.transactions:
            if transaction.sender == sender and transaction.recipient == recipient:
                return transaction

class node(object):
    def __init__(self, nodeID, addr = 0, sig = 0):
        """
        Initialization of a node in the distributed blockchain network
        :param nodeID: The specific id of the node
        :param addr: The address of the node, related to the signature
        :param sig: The signature of the node, related to the address
        """
        self.id = nodeID
        self.addr = addr
        self.sig = sig
        self.tx_pool = []
        self.own_chain = []

    def add_transaction(self, B, recipient, res, acc):
        """
        Add a transaction sent by this node
        :param B: The aim blockchain
        :param recipient: The nodeID of the recipient of the transaction
        :param res: The learning result of this node
        :param acc: The accuracy of the model trained by this node
        """
        B.new_transaction(self.id, recipient, res, acc, self.sig)

    def add_block(self, B, res, acc):
        """
        Add a block to the blockchain copy of this node
        :param B: The aim blockchain
        :param recipient: The nodeID of the recipient of the transaction
        :param res: The learning result of this node
        :param acc: The accuracy of the model trained by this node
        :return: The block
        """
        b = Block(len(B.chain) + 1, time(), self.addr, res, self.tx_pool, acc, B.hash(B.chain[-1]), self.sig)
        self.own_chain.append(b)
        self.tx_pool = []

        return b

    def get_tx_pool(self, B, spam_flag):
        """
        Get the transaction pool
        :param B: The aim blockchain
        :param spam_flag: Whether this node is a spam node
        """
        for tx in B.current_transactions:
            if tx.valid_transaction(spam_flag, get_args(tx.sender, tx.recipient)) and not tx_in_chain(self.own_chain, tx.sender, tx.recipient):
                self.tx_pool.append(tx)

    def get_chain(self, B):
        """
        Get a copy of the blockchain
        :param B: The aim blockchain
        """
        if B.valid_chain(B.chain):
            self.own_chain = B.chain

    def get_chain_PoK(self, B, neighbors):
        """
        Get a copy of the blockchain under the consensus of PoK
        :param B: The aim blockchain
        :param neighbors: The node set of all nodes in the distributed network
        """
        new_chain = None

        max_acc = self.own_chain[-1].acc

        for node in neighbors:
            if node.own_chain[-1].acc > max_acc and B.valid_chain(node.own_chain) and valid_accuracy(get_args(node.own_chain[-1].creater), node.own_chain[-1].acc):
                max_acc = node.own_chain[-1].acc
                new_chain = node.own_chain()

        if new_chain:
            self.own_chain = new_chain
            return True

        return False

    def get_chain_PoW(self, B, neighbors):
        """
        Get a copy of the blockchain under the consensus of PoW
        :param B: The aim blockchain
        :param neighbors: The node set of all nodes in the distributed network
        """
        new_chain = None

        max_length = len(self.own_chain)

        for node in neighbors:
            if len(node.own_chain) > max_length and B.valid_chain(node.own_chain):
                max_length = len(node.own_chain)
                new_chain = node.own_chain()

        if new_chain:
            self.own_chain = new_chain
            return True

        return False

def get_weights_from_ground_chain(args, GC, sender, recipient):
    # TODO: implement searching in block chain
    ground_chain_weight = tx_in_chain(GC.chain, sender, recipient).learning_result
    #weights = None
    return ground_chain_weight
def add_transaction_to_ground_chain(args, GC, sender, recipient, res=2.4, acc=0.85):
    GC.new_transaction(sender=sender, recipient=recipient, res=res, acc=acc, 0)
def add_block_to_ground_chain(args, GC, sender, res, acc):
    GC.new_block(sender=sender, res=res, acc=acc)



def get_weights_from_top_chain(args, d_matrix, D_fr):
    # TODO: implement searching in block chain
    weights = d_matrix.sum(axis=0) + D_fr
    return weights
def add_transaction_to_top_chain(args, TC, sender, recipient, res=2.4, acc=0.85):
    GC.new_transaction(sender=sender, recipient=recipient, res=res, acc=acc, 0)
def add_block_to_top_chain(args, TC, sender, res, acc):
    GC.new_block(sender=sender, res=res, acc=acc)






if __name__ == '__main__':
    # 这一部分属于我们真正在复现实验中能用到的代码……整合完了（至少注释）请删
    # Blockchain类可以是groud chain，也可以是top chain
    GC = Blockchain()
    sender = 1
    recipient = 2
    res = 2.4
    acc = 0.85
    # 添加transaction所需函数，其中作为实验模拟sender、recipient自定义，res可以设置为存储权重d_mn，acc可有可无，sig不需要
    GC.new_transaction(sender, recipient, res, acc, 0)
    # 添加block所需函数
    GC.new_block(sender, res, acc=acc)
    # 根据id查找所需transaction，并且从而得到所需权重d_mn
    print("d_mn: ", tx_in_chain(GC.chain, sender, recipient).learning_result)

    # 这两个函数是关于Fig 8的实验模拟，每一次运行代表一个block，运行多次取输出true和false的频率作为failure rate，其中第一个函数是PoK，第二个函数模拟PoW
    cnt_pok = 0
    cnt_pow = 0
    for i in range(1000):
        if GC.proof_of_knowledge_spam():
            cnt_pok = cnt_pok + 1
        if GC.proof_of_work_spam(1000 * random.random()):
            cnt_pow = cnt_pow + 1
    print("pok failure: ", cnt_pok)
    print("pow failure: ", cnt_pow)