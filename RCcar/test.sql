CREATE DATABASE board_login

USE board_login

CREATE TABLE board (
    id INT AUTO_INCREMENT PRIMARY key,
    name VARCHAR(20) NOT NULL,
    password VARCHAR(20) NOT NULL,
    subject CHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);