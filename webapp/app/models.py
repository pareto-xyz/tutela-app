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
