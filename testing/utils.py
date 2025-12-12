"""
Utilitarios para o sistema de testes colaborativos.

Inclui:
- Decoradores para retry com backoff exponencial
- Validacao de dados
- Logging configuravel
- Helpers para async operations
"""

import asyncio
import functools
import logging
import time
from typing import TypeVar, Callable, Any, Optional, Union
from urllib.parse import urlparse


# Configuracao de logging
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Retorna um logger configurado para o modulo.

    Args:
        name: Nome do logger (geralmente __name__)
        level: Nivel de logging (default: INFO)

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(f"dumont_testing.{name}")

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


# Decoradores de retry
T = TypeVar('T')


def retry_async(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorador para retry de funcoes async com backoff exponencial.

    Args:
        max_retries: Numero maximo de tentativas
        initial_delay: Delay inicial entre tentativas (segundos)
        backoff_factor: Fator multiplicador para backoff
        exceptions: Tupla de excecoes que devem triggerar retry
        on_retry: Callback opcional chamado a cada retry

    Example:
        @retry_async(max_retries=3, exceptions=(TimeoutError,))
        async def flaky_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)

                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise

            raise last_exception

        return wrapper
    return decorator


def retry_sync(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorador para retry de funcoes sync com backoff exponencial.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)

                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise

            raise last_exception

        return wrapper
    return decorator


# Validacao de dados
class ValidationError(Exception):
    """Excecao para erros de validacao."""
    pass


def validate_url(url: str, require_https: bool = False) -> str:
    """
    Valida e normaliza uma URL.

    Args:
        url: URL a validar
        require_https: Se True, requer HTTPS

    Returns:
        URL normalizada

    Raises:
        ValidationError: Se URL invalida
    """
    if not url:
        raise ValidationError("URL nao pode ser vazia")

    # Adiciona schema se nao tiver
    if not url.startswith(('http://', 'https://')):
        url = f"http://{url}"

    try:
        parsed = urlparse(url)

        if not parsed.netloc:
            raise ValidationError(f"URL invalida: {url}")

        if require_https and parsed.scheme != 'https':
            raise ValidationError(f"URL deve usar HTTPS: {url}")

        return url
    except Exception as e:
        raise ValidationError(f"URL invalida: {url} - {e}")


def validate_not_empty(value: Any, field_name: str) -> Any:
    """
    Valida que um valor nao esta vazio.

    Args:
        value: Valor a validar
        field_name: Nome do campo (para mensagem de erro)

    Returns:
        O valor validado

    Raises:
        ValidationError: Se valor vazio
    """
    if value is None:
        raise ValidationError(f"{field_name} nao pode ser None")

    if isinstance(value, str) and not value.strip():
        raise ValidationError(f"{field_name} nao pode ser string vazia")

    if isinstance(value, (list, dict)) and len(value) == 0:
        raise ValidationError(f"{field_name} nao pode ser vazio")

    return value


def validate_positive(value: Union[int, float], field_name: str) -> Union[int, float]:
    """
    Valida que um numero e positivo.

    Args:
        value: Valor a validar
        field_name: Nome do campo

    Returns:
        O valor validado

    Raises:
        ValidationError: Se valor nao positivo
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} deve ser numero, recebeu {type(value)}")

    if value <= 0:
        raise ValidationError(f"{field_name} deve ser positivo, recebeu {value}")

    return value


def validate_in_range(
    value: Union[int, float],
    field_name: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
) -> Union[int, float]:
    """
    Valida que um numero esta dentro de um range.

    Args:
        value: Valor a validar
        field_name: Nome do campo
        min_value: Valor minimo (inclusive)
        max_value: Valor maximo (inclusive)

    Returns:
        O valor validado

    Raises:
        ValidationError: Se fora do range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} deve ser numero")

    if min_value is not None and value < min_value:
        raise ValidationError(f"{field_name} deve ser >= {min_value}, recebeu {value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{field_name} deve ser <= {max_value}, recebeu {value}")

    return value


def validate_enum(value: Any, enum_class: type, field_name: str) -> Any:
    """
    Valida que um valor e membro de um Enum.

    Args:
        value: Valor a validar
        enum_class: Classe do Enum
        field_name: Nome do campo

    Returns:
        O valor do Enum

    Raises:
        ValidationError: Se valor invalido
    """
    if isinstance(value, enum_class):
        return value

    if isinstance(value, str):
        try:
            return enum_class(value)
        except ValueError:
            valid_values = [e.value for e in enum_class]
            raise ValidationError(
                f"{field_name} deve ser um de {valid_values}, recebeu '{value}'"
            )

    raise ValidationError(
        f"{field_name} deve ser {enum_class.__name__}, recebeu {type(value)}"
    )


# Helpers para async
async def run_with_timeout(
    coro,
    timeout: float,
    timeout_message: str = "Operacao excedeu timeout",
) -> Any:
    """
    Executa uma coroutine com timeout.

    Args:
        coro: Coroutine a executar
        timeout: Timeout em segundos
        timeout_message: Mensagem para TimeoutError

    Returns:
        Resultado da coroutine

    Raises:
        TimeoutError: Se exceder timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(timeout_message)


async def gather_with_errors(
    *coros,
    return_exceptions: bool = True,
) -> list[Any]:
    """
    Executa multiplas coroutines e coleta erros.

    Similar a asyncio.gather mas com melhor tratamento de erros.

    Args:
        *coros: Coroutines a executar
        return_exceptions: Se True, retorna excecoes em vez de levantar

    Returns:
        Lista de resultados (ou excecoes se return_exceptions=True)
    """
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)


# Context managers
class TimingContext:
    """Context manager para medir tempo de execucao."""

    def __init__(self, name: str = "operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
        self.duration_ms = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)

        if self.logger:
            self.logger.debug(f"{self.name} completed in {self.duration_ms}ms")

        return False


class AsyncTimingContext:
    """Context manager async para medir tempo de execucao."""

    def __init__(self, name: str = "operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
        self.duration_ms = 0

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)

        if self.logger:
            self.logger.debug(f"{self.name} completed in {self.duration_ms}ms")

        return False


# Sanitizacao
def sanitize_filename(filename: str) -> str:
    """
    Sanitiza um nome de arquivo removendo caracteres invalidos.

    Args:
        filename: Nome do arquivo

    Returns:
        Nome sanitizado
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Trunca uma string mantendo um sufixo.

    Args:
        s: String a truncar
        max_length: Comprimento maximo
        suffix: Sufixo a adicionar se truncar

    Returns:
        String truncada
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix
