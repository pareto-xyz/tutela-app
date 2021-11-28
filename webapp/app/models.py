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

    def __repr__(self) -> str:
        return f'<Address {self.address}>'


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
    meta_data: db.Column = db.Column(db.String(256))
    cluster: db.Column = db.Column(db.Integer)

    def __repr__(self) -> str:
        return f'<GasPrice {self.address}>'


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
    __table__: str = 'tornado_pool'
    id: db.Column = db.Column(db.Integer, primary_key = True)
    transaction: db.Column = db.Column(db.String(128), index = True, nullable = False)
    address: db.Column = db.Column(db.String(128), index = True, nullable = False)
    pool: db.Column = db.Column(db.String(128), index = True, nullable = False)

    def __repr__(self) -> str:
        return f'<TornadoPool {self.id}>'
