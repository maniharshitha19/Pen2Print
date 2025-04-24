document.addEventListener('DOMContentLoaded', function() {
    // Audio playback rate control
    const audioPlayer = document.getElementById('textAudio');
    const playbackRateSelect = document.getElementById('playbackRate');
    
    if (audioPlayer && playbackRateSelect) {
        playbackRateSelect.addEventListener('change', function() {
            audioPlayer.playbackRate = parseFloat(this.value);
        });
    }
    
    // Password validation for register form
    const passwordInput = document.getElementById('password');
    const confirmPasswordInput = document.getElementById('confirm_password');
    
    if (passwordInput && confirmPasswordInput) {
        function validatePassword() {
            if (passwordInput.value !== confirmPasswordInput.value) {
                confirmPasswordInput.setCustomValidity("Passwords don't match");
            } else {
                confirmPasswordInput.setCustomValidity('');
            }
        }
        
        passwordInput.addEventListener('change', validatePassword);
        confirmPasswordInput.addEventListener('keyup', validatePassword);
    }
    
    // File upload preview
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const fileName = document.querySelector('.file-name');
                if (fileName) {
                    fileName.textContent = file.name;
                }
            }
        });
    }
    
    // Flash messages auto-close
    const flashMessages = document.querySelectorAll('.flash-messages div');
    flashMessages.forEach(message => {
        setTimeout(() => {
            message.style.opacity = '0';
            setTimeout(() => message.remove(), 500);
        }, 5000);
    });
});

