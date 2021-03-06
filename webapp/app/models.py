"""Models from Database Schemas."""

from app import db


class Address(db.Model):
    __tablename__: str = 'address'
    id: db.Column = db.Column(db.Integer, primary_key = True)
    address: db.Column = db.Column(
        db.String(128),
        index = True,
        unique = True,
        nullable = False,
    )
    entity: db.Column = db.Column(db.Integer, nullable = False)
    meta_data: db.Column = db.Column(db.String(256))
    user_cluster: db.Column = db.Column(db.Integer)
    exchange_cluster: db.Column = db.Column(db.Integer)
    conf: db.Column = db.Column(db.Float)
    heuristic: db.Column = db.Column(db.Integer)

    def __repr__(self) -> str:
        return f'<Address {self.address}>'


class DepositTransaction(db.Model):
    """
    Store transactions from EOA to deposit addresses computed during DAR.
    """
    __tablename__: str = 'deposit_transaction'
    id: db.Column = db.Column(db.Integer, primary_key = True)
    address: db.Column = db.Column(
        db.String(128),
        index = True,
        nullable = False,
    )
    deposit: db.Column = db.Column(
        db.String(128),
        nullable = False,
    )
    transaction: db.Column = db.Column(
        db.String(256),
        index = True,
        unique = True,
        nullable = False,
    )
    block_number: db.Column = db.Column(db.Integer)
    block_ts: db.Column = db.Column(db.DateTime)
    conf: db.Column = db.Column(db.Float)

    def __repr__(self) -> str:
        return f'<DepositTransaction {self.address}>'


class Embedding(db.Model):
    __tablename__: str = 'embedding'
    id: db.Column = db.Column(db.Integer, primary_key = True)
    address: db.Column = db.Column(
        db.String(128),
        index = True,
        unique = True,
        nullable = False,
    )
    neighbors: db.Column = db.Column(db.String(512), nullable = False)
    distances: db.Column = db.Column(db.String(512), nullable = False)

    def __repr__(self) -> str:
        return f'<Embedding {self.address}>'


class ExactMatch(db.Model):
    __tablename__: str = 'exact_match'
    id: db.Column = db.Column(db.Integer, primary_key = True)
    address: db.Column = db.Column(
        db.String(128),
        index = True,
        nullable = False,
    )
    transaction: db.Column = db.Column(
        db.String(256),
        index = True,
        nullable = False,
    )
    block_number: db.Column = db.Column(db.Integer)
    block_ts: db.Column = db.Column(db.DateTime)
    meta_data: db.Column = db.Column(db.String(256))
    cluster: db.Column = db.Column(db.Integer)

    def __repr__(self) -> str:
        return f'<ExactMatch {self.address}>'


class GasPrice(db.Model):
    __tablename__: str = 'gas_price'
    id: db.Column = db.Column(db.Integer, primary_key = True)
    address: db.Column = db.Column(
        db.String(128),
        index = True,
        nullable = False,
    )
    transaction: db.Column = db.Column(
        db.String(256),
        index = True,
        nullable = False,
    )
    block_number: db.Column = db.Column(db.Integer)
    block_ts: db.Column = db.Column(db.DateTime)
    meta_data: db.Column = db.Column(db.String(256))
    cluster: db.Column = db.Column(db.Integer)

    def __repr__(self) -> str:
        return f'<GasPrice {self.address}>'


class MultiDenom(db.Model):
    __tablename__: str = 'multi_denom'

    id: db.Column = db.Column(db.Integer, primary_key = True)
    address: db.Column = db.Column(
        db.String(128),
        index = True,
        nullable = False,
    )
    transaction: db.Column = db.Column(
        db.String(256),
        index = True,
        nullable = False,
    )
    block_number: db.Column = db.Column(db.Integer)
    block_ts: db.Column = db.Column(db.DateTime)
    meta_data: db.Column = db.Column(db.String(256))
    cluster: db.Column = db.Column(db.Integer)

    def __repr__(self) -> str:
        return f'<MultiDenom {self.address}>'


class LinkedTransaction(db.Model):
    __tablename__: str = 'linked_transaction'

    id: db.Column = db.Column(db.Integer, primary_key = True)
    address: db.Column = db.Column(
        db.String(128),
        index = True,
        nullable = False,
    )
    transaction: db.Column = db.Column(
        db.String(256),
        index = True,
        nullable = False,
    )
    block_number: db.Column = db.Column(db.Integer)
    block_ts: db.Column = db.Column(db.DateTime)
    meta_data: db.Column = db.Column(db.String(256))
    cluster: db.Column = db.Column(db.Integer)

    def __repr__(self) -> str:
        return f'<LinkedTransaction {self.address}>'


class TornMining(db.Model):
    __tablename__: str = 'torn_mine'

    id: db.Column = db.Column(db.Integer, primary_key = True)
    address: db.Column = db.Column(
        db.String(128),
        index = True,
        nullable = False,
    )
    transaction: db.Column = db.Column(
        db.String(256),
        index = True,
        nullable = False,
    )
    block_number: db.Column = db.Column(db.Integer)
    block_ts: db.Column = db.Column(db.DateTime)
    meta_data: db.Column = db.Column(db.String(256))
    cluster: db.Column = db.Column(db.Integer)

    def __repr__(self) -> str:
        return f'<TornMining {self.address}>'


class TornadoDeposit(db.Model):
    __tablename__: str = 'tornado_deposit'
    id: db.Column = db.Column(db.Integer, primary_key = True)
    hash: db.Column = db.Column(db.String(128), index = True, nullable = False)
    transaction_index  = db.Column(db.Integer, nullable = False)
    from_address = db.Column(db.String(128), nullable = False)
    to_address = db.Column(db.String(128), nullable = False)
    gas = db.Column(db.Float)
    gas_price = db.Column(db.Float)
    block_number = db.Column(db.Integer, nullable = False)
    block_hash = db.Column(db.String(128), index = True, nullable = False)
    tornado_cash_address = db.Column(db.String(128), index = True, nullable = False)

    def __repr__(self) -> str:
        return f'<TornadoDeposit {self.hash}>'


class TornadoWithdraw(db.Model):
    __tablename__: str = 'tornado_withdraw'
    id: db.Column = db.Column(db.Integer, primary_key = True)
    hash: db.Column = db.Column(db.String(128), index = True, nullable = False)
    transaction_index  = db.Column(db.Integer, nullable = False)
    from_address = db.Column(db.String(128), nullable = False)
    to_address = db.Column(db.String(128), nullable = False)
    gas = db.Column(db.Float)
    gas_price = db.Column(db.Float)
    block_number = db.Column(db.Integer, nullable = False)
    block_hash = db.Column(db.String(128), index = True, nullable = False)
    tornado_cash_address = db.Column(db.String(128), index = True, nullable = False)
    recipient_address = db.Column(db.String(128), index = True, nullable = False)

    def __repr__(self) -> str:
        return f'<TornadoWithdraw {self.hash}>'


class TornadoPool(db.Model):
    """
    Stores an address and transaction that deposits into a tornado pool.
    """
    __tablename__: str = 'tornado_pool'
    id: db.Column = db.Column(db.Integer, primary_key = True)
    transaction: db.Column = db.Column(db.String(128), index = True, nullable = False)
    address: db.Column = db.Column(db.String(128), index = True, nullable = False)
    pool: db.Column = db.Column(db.String(128), index = True, nullable = False)

    def __repr__(self) -> str:
        return f'<TornadoPool {self.pool}>'
