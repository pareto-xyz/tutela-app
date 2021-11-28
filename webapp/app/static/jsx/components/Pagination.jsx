import React from 'react';

export default function Pagination(props) {
    return (
        <div className='all-pagination'>
            <div>
                results <span id="results-window"></span> out of <span className="total-results"></span>
            </div>
            <div className="pagination">
                <button className="page-button" id="prev-page" aria-label="Previous">
                    <span aria-hidden="true">&laquo;</span>
                </button>
                page
                <select id="page-number" className="form-select select-page" aria-label="Default select example">
                    <option selected>1</option>

                </select>
                out of <span id="total-num-pages"></span>
                <button className="page-button" id="next-page" aria-label="Next">
                    <span aria-hidden="true">&raquo;</span>
                </button>
            </div>
        </div>
    );
}