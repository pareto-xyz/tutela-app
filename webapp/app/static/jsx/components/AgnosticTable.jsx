import React from 'react';
import { Table } from 'react-bootstrap';

export default function AgnosticTable({ toIgnore, keyValues }) {
    if (toIgnore === undefined) {
        toIgnore === new Set();
    }
    return (
        <Table hover size='sm' className="detail-table">
            <tbody>
                {keyValues.map((entry, idx) => {
                    const [k, value] = entry;
                    if (toIgnore.has(k)) return;
                    return (
                        <tr className="detail-row" key={idx}>
                            <td>{k}</td>
                            <td>{value}</td>
                        </tr>)
                }
                )}
            </tbody>
        </Table>
    )
}
